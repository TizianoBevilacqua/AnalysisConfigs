from dask.distributed import WorkerPlugin, Worker, Client, get_worker

class WorkerInferenceSessionPlugin(WorkerPlugin):
    def __init__(self, model_path, session_name):
        super().__init__()
        self.model_path = model_path
        self.session_name = session_name
        self.name = f"onnx_session_{session_name}"  # stable name for lookup via worker.plugins

    async def setup(self, worker: Worker):
        import onnxruntime as ort

        sess_options = ort.SessionOptions()

        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = 1

        self.model_session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        self.input_names = [input.name for input in self.model_session.get_inputs()]
        self.output_names = [output.name for output in self.model_session.get_outputs()]


def get_model_session(model_path, model_session_name):
    try:
        worker = get_worker()
    except ValueError:
        worker = None

    if worker is None:
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = 1
        model_session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        input_name = [input.name for input in model_session.get_inputs()]
        output_name = [output.name for output in model_session.get_outputs()]

    else:
        plugin = worker.plugins[f"onnx_session_{model_session_name}"]
        model_session = plugin.model_session
        input_name = plugin.input_names
        output_name = plugin.output_names


    return model_session, input_name, output_name