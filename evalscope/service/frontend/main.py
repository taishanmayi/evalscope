import gradio as gr
import json
import os
import pandas as pd
from typing import AsyncGenerator, List, Optional, Tuple
from utils import convert_eval_args_to_config, convert_perf_args_to_config, submit_and_poll
from async_client import AsyncEvalClient

DEFAULT_SERVICE_URL = os.getenv('EVALSCOPE_SERVICE_URL', 'http://127.0.0.1:9000')
VALID_EVAL_BENCHMARKS = ['gsm8k', 'mmlu', 'cmmlu', 'ceval', 'arc', 'math_500', 'aime24', 'aime25']

# Column headers shown in the task history table
_TASK_COLUMNS = ['任务ID', '模型', '数据集', '状态', '创建时间', '完成时间']
_RESULT_COLUMNS = ['模型', '数据集', '指标', '得分', '样本数']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status_label(status: str) -> str:
    icons = {'running': '🔄 运行中', 'completed': '✅ 已完成', 'error': '❌ 失败', 'pending': '⏳ 等待中'}
    return icons.get(status, status)


def _tasks_to_df(tasks: List[dict]) -> pd.DataFrame:
    rows = []
    for t in tasks:
        datasets = t.get('datasets', [])
        if isinstance(datasets, list):
            datasets_str = ', '.join(datasets)
        else:
            datasets_str = str(datasets)
        rows.append([
            t.get('id', ''),
            t.get('model', ''),
            datasets_str,
            _status_label(t.get('status', '')),
            t.get('created_at', ''),
            t.get('completed_at', '') or '',
        ])
    return pd.DataFrame(rows, columns=_TASK_COLUMNS) if rows else pd.DataFrame(columns=_TASK_COLUMNS)


def _results_to_df(results: List[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append([
            r.get('model', ''),
            r.get('dataset', ''),
            r.get('metric', ''),
            round(float(r.get('score', 0.0)), 4),
            r.get('num', ''),
        ])
    return pd.DataFrame(rows, columns=_RESULT_COLUMNS) if rows else pd.DataFrame(columns=_RESULT_COLUMNS)


def _summary_to_pivot(rows: List[dict]) -> pd.DataFrame:
    """Build a model × dataset pivot table from summary rows."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Create a "dataset@metric" label so each column is unique
    df['dataset_metric'] = df['dataset'] + '\n' + df['metric']
    try:
        pivot = df.pivot_table(
            index='model',
            columns='dataset_metric',
            values='avg_score',
            aggfunc='mean',
        ).round(4)
        pivot = pivot.reset_index()
        pivot.columns.name = None
        return pivot
    except Exception:
        return df[['model', 'dataset', 'metric', 'avg_score']].rename(
            columns={'avg_score': '平均得分', 'model': '模型', 'dataset': '数据集', 'metric': '指标'}
        )


# ---------------------------------------------------------------------------
# Eval tab
# ---------------------------------------------------------------------------

def create_eval_interface(service_url_input, poll_interval_input, common_model_name, common_api_url, common_api_key):
    """Create the content for the evaluation task interface"""
    with gr.Row():
        # --- Left Column: Configuration (Scale 2) ---
        with gr.Column(scale=2, variant='panel'):
            gr.Markdown('### 🛠️ 评估配置')

            with gr.Accordion('评估设置', open=True):
                eval_datasets = gr.Dropdown(
                    label='数据集',
                    choices=VALID_EVAL_BENCHMARKS,
                    value=['gsm8k'],
                    multiselect=True,
                    allow_custom_value=True,
                    info='选择或输入用于评估的数据集名称，支持多选或自定义输入（逗号分隔）'
                )
                eval_limit = gr.Number(label='限制数量', value=5, precision=0, info='每个数据集的任务数量限制，-1表示不限制')
                eval_batch_size = gr.Number(label='批大小', value=1, precision=0, info='每次模型请求的批处理大小')

            with gr.Accordion('更多参数', open=False):
                eval_repeats = gr.Number(label='重复次数', value=1, precision=0, info='重复运行评估的次数')
                eval_timeout = gr.Number(label='超时时间 (秒)', value=3600, info='任务运行的最大超时时间')
                eval_stream = gr.Checkbox(label='流式输出', value=True, info='是否以流式方式获取模型响应')

                gr.Markdown('#### 模型生成参数')
                eval_temp = gr.Slider(
                    label='Temperature (随机性)', minimum=0.0, maximum=1.0, value=0.0, info='生成文本的随机性，值越大越随机'
                )
                eval_top_p = gr.Slider(
                    label='Top P (核采样)', minimum=0.0, maximum=1.0, value=1.0, info='核采样参数，只考虑累积概率达到P的词'
                )
                eval_max_tokens = gr.Number(label='Max Tokens (最大生成)', value=1024, precision=0, info='模型生成文本的最大长度')
                eval_top_k = gr.Number(label='Top K (Top K 采样)', value=50, precision=0, info='Top K 采样参数，只考虑概率最高的K个词')

                dataset_args = gr.Code(label='数据集参数 (JSON)', language='json', value='{}', lines=2, max_lines=10)

            btn_eval = gr.Button('🚀 开始评估', variant='primary', size='lg')

        # --- Right Column: Logs and Progress (Scale 3) ---
        with gr.Column(scale=3):
            gr.Markdown('### 运行状态与日志')
            eval_progress_status = gr.Markdown('当前状态: 准备就绪', label='评估任务状态')
            eval_logs = gr.Code(
                label='控制台输出', language='shell', interactive=False, lines=30, elem_classes=['log-panel'], max_lines=30
            )

    # Logic handling function
    async def run_eval_wrapper(
        service_url,
        interval,
        model,
        api_url,
        api_key,
        datasets,
        limit,
        batch_size,
        repeats,
        timeout,
        stream,
        temp,
        top_p,
        max_tokens,
        top_k,
        ds_args,
    ) -> AsyncGenerator[Tuple[str, str], None]:
        payload = convert_eval_args_to_config(
            model=model,
            api_url=api_url,
            api_key=api_key,
            datasets=datasets,
            limit=limit,
            eval_batch_size=batch_size,
            repeats=repeats,
            timeout=timeout,
            stream=stream,
            temperature=temp,
            top_p=top_p,
            max_tokens=max_tokens,
            top_k=top_k,
            dataset_args=ds_args
        )
        async for log_content, progress_status_text in submit_and_poll(service_url, 'eval', payload, interval):
            yield log_content, progress_status_text

    btn_eval.click(
        run_eval_wrapper,
        inputs=[
            service_url_input, poll_interval_input, common_model_name, common_api_url, common_api_key, eval_datasets,
            eval_limit, eval_batch_size, eval_repeats, eval_timeout, eval_stream, eval_temp, eval_top_p,
            eval_max_tokens, eval_top_k, dataset_args
        ],
        outputs=[eval_logs, eval_progress_status]
    )


# ---------------------------------------------------------------------------
# Perf tab
# ---------------------------------------------------------------------------

def create_perf_interface(service_url_input, poll_interval_input, common_model_name, common_api_url, common_api_key):
    """Create the content for the performance testing interface"""
    with gr.Row():
        # --- Left Column: Configuration (Scale 2) ---
        with gr.Column(scale=2, variant='panel'):
            gr.Markdown('### ⚡ 性能测试配置')

            with gr.Accordion('压测设置', open=True):
                perf_api_type = gr.Dropdown(
                    label='API类型', choices=['openai'], value='openai', info='指定API接口类型，目前支持OpenAI兼容接口'
                )
                perf_parallel = gr.Textbox(
                    label='并发数 (逗号分隔)', value='1', placeholder='例如: 1,2,4', info='逗号分隔的并发用户数列表，可定义多个并发等级'
                )
                perf_number = gr.Textbox(
                    label='总请求数 (逗号分隔)', value='10', placeholder='例如: 10,20', info='逗号分隔的总请求数列表，对应每个并发等级的总请求数'
                )
                perf_rate = gr.Number(label='速率限制 (请求/秒)', value=-1, precision=0, info='-1 表示不限制每秒请求数')

            with gr.Accordion('模型生成参数', open=False):
                perf_max_tokens = gr.Number(label='Max Tokens (最大生成)', value=2048, precision=0, info='模型生成文本的最大长度')
                perf_min_tokens = gr.Number(label='Min Tokens (最小生成)', value=0, precision=0, info='模型生成文本的最小长度')

                perf_temp = gr.Slider(
                    label='Temperature (随机性)', minimum=0.0, maximum=1.0, value=0.0, info='生成文本的随机性，值越大越随机'
                )
                perf_top_p = gr.Slider(
                    label='Top P (核采样)', minimum=0.0, maximum=1.0, value=1.0, info='核采样参数，只考虑累积概率达到P的词'
                )

            with gr.Accordion('数据集设置', open=False):
                perf_dataset = gr.Dropdown(
                    label='测试数据集', choices=['openqa', 'line_by_line', 'random'], value='openqa', info='用于性能测试的数据集类型'
                )

                perf_max_prompt = gr.Number(
                    label='Max Prompt Len (最大Prompt长度)', value=1024, precision=0, info='生成Prompt的最大长度'
                )
                perf_min_prompt = gr.Number(
                    label='Min Prompt Len (最小Prompt长度)', value=0, precision=0, info='生成Prompt的最小长度'
                )

            btn_perf = gr.Button('⚡ 开始性能测试', variant='primary', size='lg')

        # --- Right Column: Logs and Progress (Scale 3) ---
        with gr.Column(scale=3):
            gr.Markdown('### 运行状态与日志')
            perf_progress_status = gr.Markdown('当前状态: 准备就绪', label='性能测试任务状态')
            perf_logs = gr.Code(
                label='控制台输出', language='shell', interactive=False, lines=30, elem_classes=['log-panel'], max_lines=30
            )

    # Logic handling function
    async def run_perf_wrapper(
        service_url,
        interval,
        model,
        url,
        api_key,
        api_type,
        parallel,
        number,
        rate,
        max_tokens,
        min_tokens,
        temp,
        top_p,
        dataset,
        max_pl,
        min_pl,
    ) -> AsyncGenerator[Tuple[str, str], None]:
        payload = convert_perf_args_to_config(
            model=model,
            url=url,
            api=api_type,
            api_key=api_key,
            parallel=parallel,
            number=number,
            rate=rate,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            temperature=temp,
            top_p=top_p,
            dataset=dataset,
            max_prompt_length=max_pl,
            min_prompt_length=min_pl
        )
        async for log_content, progress_status_text in submit_and_poll(service_url, 'perf', payload, interval):
            yield log_content, progress_status_text

    btn_perf.click(
        run_perf_wrapper,
        inputs=[
            service_url_input, poll_interval_input, common_model_name, common_api_url, common_api_key, perf_api_type,
            perf_parallel, perf_number, perf_rate, perf_max_tokens, perf_min_tokens, perf_temp, perf_top_p,
            perf_dataset, perf_max_prompt, perf_min_prompt
        ],
        outputs=[perf_logs, perf_progress_status]
    )


# ---------------------------------------------------------------------------
# Task History tab
# ---------------------------------------------------------------------------

def create_history_interface(service_url_input):
    """Create the Task History tab — lists all past evaluation tasks."""

    # State for the currently selected task ID
    selected_task_id = gr.State(value='')

    with gr.Row():
        # ---- Left: Filters ----
        with gr.Column(scale=1, variant='panel'):
            gr.Markdown('### 🔍 筛选')
            status_filter = gr.Dropdown(
                label='状态筛选',
                choices=['全部', 'running', 'completed', 'error', 'pending'],
                value='全部',
                info='按任务状态筛选'
            )
            limit_input = gr.Number(label='最多显示', value=100, precision=0, minimum=1, maximum=500)
            btn_refresh = gr.Button('🔄 刷新列表', variant='secondary')
            btn_delete = gr.Button('🗑️ 删除选中任务', variant='stop')
            delete_status = gr.Markdown('')

        # ---- Right: Table + Detail panel ----
        with gr.Column(scale=4):
            gr.Markdown('### 📋 评测任务列表')
            task_table = gr.Dataframe(
                headers=_TASK_COLUMNS,
                datatype=['str'] * len(_TASK_COLUMNS),
                interactive=False,
                wrap=True,
                label='点击任意行查看详情',
            )

            with gr.Accordion('📊 任务详情', open=False) as detail_accordion:
                with gr.Row():
                    task_id_display = gr.Textbox(label='任务ID', interactive=False)
                    task_status_display = gr.Textbox(label='状态', interactive=False)

                task_error_display = gr.Textbox(label='错误信息', interactive=False, visible=False)
                task_config_display = gr.Code(label='任务配置 (JSON)', language='json', interactive=False, lines=8)

                gr.Markdown('#### 评测结果')
                results_table = gr.Dataframe(
                    headers=_RESULT_COLUMNS,
                    datatype=['str', 'str', 'str', 'number', 'number'],
                    interactive=False,
                    label='指标得分',
                )

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def refresh_tasks(service_url: str, status: str, limit: int):
        status_val = None if status == '全部' else status
        async with AsyncEvalClient(service_url) as client:
            tasks = await client.get_tasks(status=status_val, limit=int(limit))
        return _tasks_to_df(tasks), gr.update(open=False), '', pd.DataFrame(columns=_RESULT_COLUMNS)

    async def on_row_select(evt: gr.SelectData, task_df: pd.DataFrame, service_url: str):
        """When a row is clicked in the task table, load and display its details."""
        if task_df is None or task_df.empty:
            return '', '', False, '', '{}', pd.DataFrame(columns=_RESULT_COLUMNS), gr.update(open=False)

        row_idx = evt.index[0]
        if row_idx >= len(task_df):
            return '', '', False, '', '{}', pd.DataFrame(columns=_RESULT_COLUMNS), gr.update(open=False)

        task_id = str(task_df.iloc[row_idx]['任务ID'])

        async with AsyncEvalClient(service_url) as client:
            detail = await client.get_task_detail(task_id)

        if not detail:
            return task_id, '未找到', False, '', '{}', pd.DataFrame(columns=_RESULT_COLUMNS), gr.update(open=True)

        status = detail.get('status', '')
        error = detail.get('error') or ''
        config = detail.get('config', {})
        results = detail.get('results', [])

        config_str = json.dumps(config, indent=2, ensure_ascii=False) if isinstance(config, dict) else str(config)
        results_df = _results_to_df(results)

        return (
            task_id,
            _status_label(status),
            bool(error),
            error,
            config_str,
            results_df,
            gr.update(open=True),
        )

    async def delete_selected(service_url: str, task_id: str, status: str, limit: int):
        if not task_id:
            return '⚠️ 请先点击选择一个任务', pd.DataFrame(columns=_TASK_COLUMNS), ''
        async with AsyncEvalClient(service_url) as client:
            ok = await client.delete_task(task_id)
        if ok:
            msg = f'✅ 任务 `{task_id}` 已删除'
            # Refresh the list
            status_val = None if status == '全部' else status
            async with AsyncEvalClient(service_url) as client:
                tasks = await client.get_tasks(status=status_val, limit=int(limit))
            return msg, _tasks_to_df(tasks), ''
        else:
            return f'❌ 删除失败: `{task_id}`', pd.DataFrame(columns=_TASK_COLUMNS), ''

    # Wire up events
    btn_refresh.click(
        refresh_tasks,
        inputs=[service_url_input, status_filter, limit_input],
        outputs=[task_table, detail_accordion, task_id_display, results_table],
    )

    task_table.select(
        on_row_select,
        inputs=[task_table, service_url_input],
        outputs=[
            task_id_display,
            task_status_display,
            task_error_display,  # visible
            task_error_display,  # value
            task_config_display,
            results_table,
            detail_accordion,
        ],
    )

    btn_delete.click(
        delete_selected,
        inputs=[service_url_input, task_id_display, status_filter, limit_input],
        outputs=[delete_status, task_table, task_id_display],
    )

    return selected_task_id


# ---------------------------------------------------------------------------
# Results Summary tab
# ---------------------------------------------------------------------------

def create_summary_interface(service_url_input):
    """Create the Results Summary tab — aggregated scores across all tasks."""

    with gr.Row():
        # ---- Left: Filters ----
        with gr.Column(scale=1, variant='panel'):
            gr.Markdown('### 🔍 筛选')
            model_filter = gr.Textbox(label='模型名称（模糊匹配）', placeholder='例如: qwen', value='')
            dataset_filter = gr.Textbox(label='数据集名称（模糊匹配）', placeholder='例如: gsm8k', value='')
            btn_summary_refresh = gr.Button('🔄 刷新汇总', variant='secondary')
            csv_download = gr.File(label='导出 CSV', visible=False)
            btn_export = gr.Button('📥 导出 CSV', variant='secondary')

        # ---- Right: Pivot table ----
        with gr.Column(scale=4):
            gr.Markdown('### 📈 评测结果汇总 (模型 × 数据集)')
            gr.Markdown(
                '_每格显示该模型在该数据集上的平均得分（跨所有已完成任务）。_',
                elem_id='summary-desc',
            )
            summary_pivot = gr.Dataframe(
                interactive=False,
                wrap=True,
                label='汇总表格（模型为行，数据集@指标为列）',
            )
            gr.Markdown('### 📋 详细汇总')
            summary_detail = gr.Dataframe(
                headers=['模型', '数据集', '指标', '平均得分', '最高得分', '最低得分', '运行次数'],
                datatype=['str', 'str', 'str', 'number', 'number', 'number', 'number'],
                interactive=False,
                label='各模型/数据集/指标统计',
            )

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def refresh_summary(service_url: str, model: str, dataset: str):
        async with AsyncEvalClient(service_url) as client:
            rows = await client.get_summary(
                model=model.strip() or None,
                dataset=dataset.strip() or None,
            )
        pivot_df = _summary_to_pivot(rows)

        # Build detail df
        if rows:
            detail_rows = [
                [
                    r['model'], r['dataset'], r['metric'],
                    round(r['avg_score'], 4),
                    round(r['max_score'], 4),
                    round(r['min_score'], 4),
                    r['run_count'],
                ]
                for r in rows
            ]
            detail_df = pd.DataFrame(
                detail_rows,
                columns=['模型', '数据集', '指标', '平均得分', '最高得分', '最低得分', '运行次数'],
            )
        else:
            detail_df = pd.DataFrame(
                columns=['模型', '数据集', '指标', '平均得分', '最高得分', '最低得分', '运行次数']
            )

        return pivot_df, detail_df

    async def export_csv(service_url: str, model: str, dataset: str):
        async with AsyncEvalClient(service_url) as client:
            rows = await client.get_summary(
                model=model.strip() or None,
                dataset=dataset.strip() or None,
            )
        if not rows:
            return gr.update(visible=False)

        df = pd.DataFrame(rows)
        csv_path = '/tmp/evalscope_summary.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        return gr.update(value=csv_path, visible=True)

    btn_summary_refresh.click(
        refresh_summary,
        inputs=[service_url_input, model_filter, dataset_filter],
        outputs=[summary_pivot, summary_detail],
    )

    btn_export.click(
        export_csv,
        inputs=[service_url_input, model_filter, dataset_filter],
        outputs=[csv_download],
    )


# ---------------------------------------------------------------------------
# Main interface assembly
# ---------------------------------------------------------------------------

def create_interface():
    with gr.Blocks(title='EvalScope Dashboard', theme=gr.themes.Soft()) as demo:
        gr.Markdown('# 🚀 EvalScope 服务面板')

        # Global Service Settings (Top Bar)
        with gr.Accordion('全局设置', open=True):
            with gr.Row():
                service_url_input = gr.Textbox(
                    label='EvalScope 服务URL', value=DEFAULT_SERVICE_URL, scale=3, info='EvalScope后端服务的访问地址'
                )
                poll_interval_input = gr.Number(label='日志轮询间隔 (秒)', value=5, minimum=5, scale=1, info='获取任务日志的间隔时间')

            with gr.Row():
                common_model_name = gr.Textbox(
                    label='模型名称', value='qwen-plus', placeholder='例如: qwen-max, gpt-4', scale=1, info='用于评估或性能测试的模型名称'
                )
                common_api_url = gr.Textbox(
                    label='模型API URL',
                    value='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
                    scale=2,
                    info='模型API的请求地址'
                )
                common_api_key = gr.Textbox(
                    label='模型API Key',
                    value=os.getenv('DASHSCOPE_API_KEY', ''),
                    type='password',
                    scale=1,
                    info='访问模型API所需的密钥'
                )

        with gr.Tabs():
            with gr.TabItem('📊 模型评估'):
                create_eval_interface(
                    service_url_input, poll_interval_input, common_model_name, common_api_url, common_api_key
                )

            with gr.TabItem('⚡ 性能测试'):
                create_perf_interface(
                    service_url_input, poll_interval_input, common_model_name, common_api_url, common_api_key
                )

            with gr.TabItem('📋 任务历史'):
                create_history_interface(service_url_input)

            with gr.TabItem('📈 结果汇总'):
                create_summary_interface(service_url_input)

    return demo


if __name__ == '__main__':
    demo = create_interface()
    demo.queue().launch()
