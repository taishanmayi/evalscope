# Copyright (c) Alibaba, Inc. and its affiliates.
"""Task-management REST endpoints backed by the SQLite task database."""

from flask import Blueprint, jsonify, request

from evalscope.utils.logger import get_logger
from ..db import get_db
from ..utils import validate_task_id

logger = get_logger()

bp_tasks = Blueprint('tasks', __name__, url_prefix='/api/v1/tasks')


# ---------------------------------------------------------------------------
# GET /api/v1/tasks
# ---------------------------------------------------------------------------

@bp_tasks.route('', methods=['GET'])
def list_tasks():
    """List evaluation tasks stored in the database.

    Query params:
        status (str, optional): Filter by status (pending/running/completed/error).
        limit  (int, optional): Max records to return (default 100).
        offset (int, optional): Skip this many records (default 0).
    """
    status = request.args.get('status')
    limit = request.args.get('limit', 100, type=int)
    offset = request.args.get('offset', 0, type=int)

    try:
        tasks = get_db().get_tasks(status=status, limit=limit, offset=offset)
        return jsonify({'tasks': tasks, 'count': len(tasks)}), 200
    except Exception as e:
        logger.error(f'Failed to list tasks: {e}')
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# GET /api/v1/tasks/summary
# ---------------------------------------------------------------------------

@bp_tasks.route('/summary', methods=['GET'])
def get_summary():
    """Return aggregated scores grouped by model × dataset × metric.

    Query params:
        model   (str, optional): Substring filter on model name.
        dataset (str, optional): Substring filter on dataset name.
    """
    model = request.args.get('model') or None
    dataset = request.args.get('dataset') or None

    try:
        rows = get_db().get_summary(model=model, dataset=dataset)
        return jsonify({'summary': rows, 'count': len(rows)}), 200
    except Exception as e:
        logger.error(f'Failed to get summary: {e}')
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# GET /api/v1/tasks/<task_id>
# ---------------------------------------------------------------------------

@bp_tasks.route('/<task_id>', methods=['GET'])
def get_task(task_id: str):
    """Return a single task record together with its metric results.

    Path params:
        task_id (str): The task identifier.
    """
    try:
        validate_task_id(task_id)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    try:
        task = get_db().get_task(task_id)
        if task is None:
            return jsonify({'error': f'Task not found: {task_id}'}), 404

        results = get_db().get_task_results(task_id)
        task['results'] = results
        return jsonify(task), 200
    except Exception as e:
        logger.error(f'Failed to get task {task_id}: {e}')
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# DELETE /api/v1/tasks/<task_id>
# ---------------------------------------------------------------------------

@bp_tasks.route('/<task_id>', methods=['DELETE'])
def delete_task(task_id: str):
    """Delete a task record (and its associated results) from the database.

    Path params:
        task_id (str): The task identifier.
    """
    try:
        validate_task_id(task_id)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    try:
        db = get_db()
        if db.get_task(task_id) is None:
            return jsonify({'error': f'Task not found: {task_id}'}), 404
        db.delete_task(task_id)
        return jsonify({'status': 'deleted', 'task_id': task_id}), 200
    except Exception as e:
        logger.error(f'Failed to delete task {task_id}: {e}')
        return jsonify({'error': str(e)}), 500
