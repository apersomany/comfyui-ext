def render_json(result):
    return result.model_dump_json(indent=2)
