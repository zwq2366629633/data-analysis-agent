import os
import json
from io import BytesIO
import streamlit as st
import pandas as pd
import duckdb
import sqlparse
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# 1. 读取 .env 文件里的配置
# =========================

load_dotenv()

def get_secret(name, default=None):
    value = os.getenv(name)
    if value:
        return value

    try:
        return st.secrets[name]
    except Exception:
        return default


api_key = get_secret("DEEPSEEK_API_KEY")
model_name = get_secret("MODEL_NAME", "deepseek-chat")

if api_key:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
else:
    client = None


# =========================
# 2. 页面基础信息
# =========================

st.title("智能数据分析助手")

st.write("上传 CSV 文件，输入中文数据问题，系统会调用 DeepSeek 自动生成 SQL、查询数据，并输出分析结论。")


# =========================
# 3. 上传 CSV 文件
# =========================

st.subheader("上传数据文件")

uploaded_file = st.file_uploader(
    "请上传 CSV 或 Excel 文件。如果不上传，则默认使用 data/sales.csv 示例数据。",
    type=["csv", "xlsx", "xls"]
)


def read_uploaded_file(uploaded_file):
    """
    读取用户上传的 CSV / Excel 文件。
    支持：
    - CSV：.csv
    - Excel：.xlsx / .xls
    """
    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    # 读取 CSV 文件
    if file_name.endswith(".csv"):
        encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]

        for enc in encodings:
            try:
                return pd.read_csv(BytesIO(file_bytes), encoding=enc), None
            except UnicodeDecodeError:
                continue

        # 如果常见编码都失败，最后尝试 latin1
        return pd.read_csv(BytesIO(file_bytes), encoding="latin1"), None

    # 读取 Excel 文件
    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        excel_file = pd.ExcelFile(BytesIO(file_bytes))
        sheet_names = excel_file.sheet_names

        if len(sheet_names) > 1:
            selected_sheet = st.selectbox(
                "检测到多个工作表，请选择要分析的工作表：",
                sheet_names
            )
        else:
            selected_sheet = sheet_names[0]

        df_excel = pd.read_excel(
            BytesIO(file_bytes),
            sheet_name=selected_sheet
        )

        return df_excel, selected_sheet

    else:
        st.error("暂时只支持 CSV、XLSX、XLS 文件。")
        st.stop()


if uploaded_file is not None:
    df, selected_sheet = read_uploaded_file(uploaded_file)

    if selected_sheet is not None:
        data_source = f"用户上传 Excel 文件：{uploaded_file.name}，工作表：{selected_sheet}"
    else:
        data_source = f"用户上传 CSV 文件：{uploaded_file.name}"

else:
    df = pd.read_csv("data/sales.csv")
    data_source = "默认示例数据：data/sales.csv"


st.info(f"当前数据来源：{data_source}")


# =========================
# 4. 展示原始数据
# =========================

st.subheader("原始数据预览")

st.write(f"数据共有 {df.shape[0]} 行，{df.shape[1]} 列。")

st.dataframe(df.head(100), use_container_width=True)


# =========================
# 5. 自动生成数据结构说明
# =========================

def build_schema_info(df: pd.DataFrame) -> str:
    """
    根据上传的数据自动生成字段说明，交给大模型使用。
    """
    column_info = []

    for col in df.columns:
        dtype = str(df[col].dtype)

        # 取前几个非空样例值
        sample_values = (
            df[col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()[:5]
        )

        column_info.append(
            f"- {col}：数据类型 {dtype}，样例值 {sample_values}"
        )

    schema_text = "\n".join(column_info)

    return f"""
当前只有一张表，表名是 df。

字段信息如下：
{schema_text}

重要规则：
1. SQL 只能查询 df 表。
2. SQL 只能是 SELECT 查询。
3. 不允许 INSERT、UPDATE、DELETE、DROP、ALTER、CREATE、TRUNCATE。
4. 如果字段名包含中文、空格或特殊符号，SQL 中必须用英文双引号包住字段名。
   例如：
   SELECT SUM("销售额") FROM df
5. 不要使用不存在的字段。
6. 如果用户问题无法根据当前字段回答，请生成一个简单的 SELECT 语句查看相关字段，不要编造字段。
"""


SCHEMA_INFO = build_schema_info(df)


with st.expander("查看系统识别到的数据字段"):
    st.text(SCHEMA_INFO)


# =========================
# 6. 展示整体核心指标
# 如果是默认 sales.csv，就展示固定业务指标
# 如果是用户上传的其他数据，就只展示基础信息
# =========================

st.subheader("数据概览")

col1, col2, col3 = st.columns(3)

col1.metric("行数", f"{df.shape[0]:,}")
col2.metric("列数", f"{df.shape[1]:,}")
col3.metric("缺失值数量", f"{df.isna().sum().sum():,}")
# =========================
# 数据质量与统计分析模块
# =========================

st.subheader("数据质量与统计分析")

# 1. 缺失值分析
st.markdown("### 1. 缺失值分析")

missing_df = pd.DataFrame({
    "字段名": df.columns,
    "缺失值数量": df.isna().sum().values,
    "缺失率": (df.isna().sum().values / len(df))
})

missing_df["缺失率"] = missing_df["缺失率"].apply(lambda x: f"{x:.2%}")

st.dataframe(missing_df, use_container_width=True)

# 2. 数值型字段描述性统计
st.markdown("### 2. 数值型字段描述性统计")

numeric_df = df.select_dtypes(include="number")

if numeric_df.empty:
    st.info("当前数据中没有数值型字段，无法进行描述性统计。")
else:
    desc_df = numeric_df.describe().T

    desc_df = desc_df.rename(columns={
        "count": "样本数",
        "mean": "均值",
        "std": "标准差",
        "min": "最小值",
        "25%": "25%分位数",
        "50%": "中位数",
        "75%": "75%分位数",
        "max": "最大值"
    })

    st.dataframe(desc_df, use_container_width=True)

# 3. 异常值检测：IQR 方法
st.markdown("### 3. 异常值检测")

if numeric_df.empty:
    st.info("当前数据中没有数值型字段，无法进行异常值检测。")
else:
    outlier_summary = []

    for col in numeric_df.columns:
        q1 = numeric_df[col].quantile(0.25)
        q3 = numeric_df[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = numeric_df[
            (numeric_df[col] < lower_bound) | 
            (numeric_df[col] > upper_bound)
        ].shape[0]

        outlier_summary.append({
            "字段名": col,
            "下界": lower_bound,
            "上界": upper_bound,
            "异常值数量": outlier_count,
            "异常值占比": outlier_count / len(df)
        })

    outlier_df = pd.DataFrame(outlier_summary)
    outlier_df["异常值占比"] = outlier_df["异常值占比"].apply(lambda x: f"{x:.2%}")

    st.dataframe(outlier_df, use_container_width=True)

# 4. 数据质量总结
st.markdown("### 4. 数据质量总结")

total_missing = df.isna().sum().sum()
total_cells = df.shape[0] * df.shape[1]
missing_rate = total_missing / total_cells if total_cells > 0 else 0

if missing_rate == 0:
    st.success("当前数据没有缺失值，数据完整性较好。")
elif missing_rate < 0.05:
    st.warning(f"当前数据整体缺失率为 {missing_rate:.2%}，缺失情况较轻。")
else:
    st.error(f"当前数据整体缺失率为 {missing_rate:.2%}，建议先处理缺失值后再分析。")

# 如果数据里刚好有这些字段，就展示电商核心指标
required_cols = {"gmv", "orders", "visitors", "refund_amount"}

if required_cols.issubset(set(df.columns)):
    st.subheader("电商核心指标")

    total_gmv = df["gmv"].sum()
    total_orders = df["orders"].sum()
    total_visitors = df["visitors"].sum()
    total_refund = df["refund_amount"].sum()

    conversion_rate = total_orders / total_visitors if total_visitors != 0 else 0
    refund_rate = total_refund / total_gmv if total_gmv != 0 else 0

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("总 GMV", f"{total_gmv:,.0f} 元")
    m2.metric("总订单数", f"{total_orders:,.0f}")
    m3.metric("整体转化率", f"{conversion_rate:.2%}")
    m4.metric("整体退款率", f"{refund_rate:.2%}")


# =========================
# 7. 调用 DeepSeek 生成 SQL
# =========================

def generate_sql(question: str) -> str:
    prompt = f"""
你是一个严谨的数据分析 SQL 助手。

请根据用户问题，生成 DuckDB 可以执行的 SQL。

必须遵守：
1. 只能生成 SELECT 查询。
2. 只能查询 df 表。
3. 不能使用 INSERT、UPDATE、DELETE、DROP、ALTER、CREATE、TRUNCATE。
4. 必须严格根据字段说明生成 SQL。
5. 不要使用不存在的字段。
6. 如果字段名包含中文、空格或特殊符号，必须用英文双引号包住字段名。
7. 只返回 JSON，不要返回 Markdown，不要写 ```json。
8. JSON 格式如下：

{{
  "sql": "SELECT ...",
  "reason": "解释为什么这样写 SQL"
}}

数据表说明：
{SCHEMA_INFO}

用户问题：
{question}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你是严谨的数据分析 SQL 助手，只输出 JSON。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    # 防止模型偶尔返回 ```json 包裹
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(content)
        return result["sql"]
    except Exception:
        st.error("DeepSeek 返回的内容不是合法 JSON。原始返回如下：")
        st.code(content)
        st.stop()


# =========================
# 8. SQL 安全检查
# =========================

def is_safe_sql(sql: str):
    sql_lower = sql.strip().lower()

    forbidden_words = [
        "insert", "update", "delete", "drop", "alter",
        "create", "truncate", "replace", "grant", "revoke"
    ]

    if not sql_lower.startswith("select"):
        return False, "只允许执行 SELECT 查询。"

    for word in forbidden_words:
        if word in sql_lower:
            return False, f"SQL 中包含危险操作：{word}"

    parsed = sqlparse.parse(sql)

    if len(parsed) != 1:
        return False, "不允许一次执行多条 SQL。"

    if "from df" not in sql_lower:
        return False, "SQL 只能查询 df 表。"

    return True, "SQL 安全检查通过。"


# =========================
# 9. 调用 DeepSeek 生成分析结论
# =========================

def analyze_result(question: str, sql: str, result_text: str) -> str:
    prompt = f"""
你是一个业务数据分析师。

用户问题：
{question}

执行的 SQL：
{sql}

查询结果：
{result_text}

请基于查询结果，用中文输出分析结论。

输出格式：
1. 核心结论
2. 关键数据
3. 异常或值得关注的点
4. 建议动作

要求：
- 只能基于查询结果分析。
- 不要编造查询结果里没有的数据。
- 如果数据不足，要明确说明。
- 语言要适合写进业务分析报告。
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你是严谨的业务数据分析师。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# =========================
# 10. 用户输入问题
# =========================

st.subheader("请输入你的数据问题")

question = st.text_input(
    "例如：各渠道 GMV 排名怎么样？哪个地区销售额最高？按日期统计趋势？",
    placeholder="请输入你的问题"
)

if st.button("开始分析"):

    if not api_key:
        st.error("没有读取到 DEEPSEEK_API_KEY，请检查 .env 文件。")
        st.stop()

    if question.strip() == "":
        st.warning("请先输入一个问题。")
        st.stop()

    with st.spinner("DeepSeek 正在生成 SQL..."):
        sql = generate_sql(question)

    st.subheader("DeepSeek 生成的 SQL")
    st.code(sql, language="sql")

    safe, message = is_safe_sql(sql)

    if not safe:
        st.error(message)
        st.stop()

    st.success(message)

    try:
        with st.spinner("正在查询数据..."):
            result = duckdb.sql(sql).df()
    except Exception as e:
        st.error("SQL 执行失败，可能是字段名、语法或数据类型问题。")
        st.code(str(e))
        st.stop()

    st.subheader("查询结果")
    st.dataframe(result, use_container_width=True)

    if result.empty:
        st.warning("查询结果为空，无法生成分析结论。")
        st.stop()

    # 自动图表
    st.subheader("自动图表")

    numeric_columns = result.select_dtypes(include="number").columns.tolist()
    non_numeric_columns = result.select_dtypes(exclude="number").columns.tolist()

    if len(numeric_columns) > 0 and len(non_numeric_columns) > 0:
        x_col = non_numeric_columns[0]
        y_col = numeric_columns[0]

        chart_data = result.set_index(x_col)[y_col]
        st.bar_chart(chart_data)
    else:
        st.info("当前查询结果不适合自动生成柱状图。")

    # 生成分析结论
    result_text = result.to_string(index=False)

    with st.spinner("DeepSeek 正在生成分析结论..."):
        analysis = analyze_result(question, sql, result_text)

    st.subheader("分析结论")
    st.markdown(analysis)


# =========================
# 11. 如果存在日期和 GMV 字段，展示默认趋势图
# =========================

if "date" in df.columns and "gmv" in df.columns:
    st.subheader("每日 GMV 趋势")

    daily_sql = """
    SELECT
        date,
        SUM(gmv) AS total_gmv
    FROM df
    GROUP BY date
    ORDER BY date
    """

    daily_result = duckdb.sql(daily_sql).df()

    st.line_chart(daily_result.set_index("date")["total_gmv"])


# =========================
# 12. 一键生成经营分析报告
# =========================

st.subheader("一键生成数据分析报告")

if st.button("生成数据分析报告"):

    # 如果是电商数据，生成电商经营报告
    if required_cols.issubset(set(df.columns)) and "channel" in df.columns:
        report_sql = """
        SELECT
            channel,
            SUM(gmv) AS total_gmv,
            SUM(orders) AS total_orders,
            SUM(visitors) AS total_visitors,
            SUM(orders) * 1.0 / SUM(visitors) AS conversion_rate,
            SUM(refund_amount) * 1.0 / SUM(gmv) AS refund_rate
        FROM df
        GROUP BY channel
        ORDER BY total_gmv DESC
        """

        report_question = """
        请根据各渠道经营数据，生成一份电商经营分析报告。
        报告需要包括：
        1. 核心结论
        2. 各渠道表现分析
        3. 转化率和退款率分析
        4. 异常或值得关注的点
        5. 后续运营建议
        """

    else:
        # 如果用户上传的是其他 CSV，就生成通用数据概览报告
        report_sql = """
        SELECT *
        FROM df
        LIMIT 50
        """

        report_question = """
        请根据当前 CSV 数据样例，生成一份通用数据分析报告。
        报告需要包括：
        1. 数据内容概览
        2. 主要字段解释
        3. 初步发现
        4. 可能的分析方向
        5. 后续建议
        """

    with st.spinner("正在准备报告数据..."):
        report_result = duckdb.sql(report_sql).df()

    st.subheader("报告数据表")
    st.dataframe(report_result, use_container_width=True)

    report_text = report_result.to_string(index=False)

    with st.spinner("DeepSeek 正在生成数据分析报告..."):
        report = analyze_result(
            report_question,
            report_sql,
            report_text
        )

    st.subheader("数据分析报告")
    st.markdown(report)

    st.download_button(
        label="下载数据分析报告",
        data=report,
        file_name="数据分析报告.md",
        mime="text/markdown"
    )
