from unittest import mock

from rambla.utils.mlflow import log_artifacts, mlflow_log


@mock.patch("rambla.utils.mlflow.mlflow.log_artifact")
@mock.patch("rambla.utils.mlflow.dump")
def test_log_artifacts_single_call(mock_dump, mock_mlflow_log_artifact, tmpdir):
    dummy_dict = {"input_0": 0, "input_1": "1"}

    with mock.patch("os.path.expanduser") as mock_expanduser:
        mock_expanduser.return_value = tmpdir
        log_artifacts(extension="json", dummy=dummy_dict)

    #
    mock_dump.assert_called_once()
    assert mock_dump.call_args[0][0] == dummy_dict
    assert mock_dump.call_args[0][1].name == "dummy.json"

    mock_mlflow_log_artifact.assert_called_once()
    assert mock_mlflow_log_artifact.call_args[0][0].name == "dummy.json"


@mock.patch("rambla.utils.mlflow.mlflow.log_artifact")
@mock.patch("rambla.utils.mlflow.dump")
def test_log_artifacts_two_calls(mock_dump, mock_mlflow_log_artifact, tmpdir):
    dummy_dict = {"input_0": 0, "input_1": "1"}
    dummy_list = [1, 2, 3]

    with mock.patch("os.path.expanduser") as mock_expanduser:
        mock_expanduser.return_value = tmpdir
        log_artifacts(extension="json", dummy_dict=dummy_dict, dummy_list=dummy_list)

    #
    assert mock_dump.call_count == 2
    assert mock_dump.call_args_list[0][0][0] == dummy_dict
    assert mock_dump.call_args_list[0][0][1].name == "dummy_dict.json"
    assert mock_dump.call_args_list[1][0][0] == dummy_list
    assert mock_dump.call_args_list[1][0][1].name == "dummy_list.json"

    assert mock_mlflow_log_artifact.call_count == 2
    assert mock_mlflow_log_artifact.call_args_list[0][0][0].name == "dummy_dict.json"
    assert mock_mlflow_log_artifact.call_args_list[1][0][0].name == "dummy_list.json"


@mock.patch("rambla.utils.mlflow.log_artifacts")
@mock.patch("rambla.utils.mlflow.mlflow.log_metrics")
@mock.patch("rambla.utils.mlflow.mlflow.log_params")
def test_mlflow_log(
    mock_log_params,
    mock_log_metrics,
    mock_log_artifacts,
):
    project_name = "mock_project_name"
    experiment_name = "mock_experiment_name"
    run_name = "mock_run_name"

    config = {"mock_config": "a"}
    artifacts = {"mock_artifacts": [1, 2, 3]}
    metrics = {"mock_metrics": 3.0}

    extension = "pkl"

    mlflow_log(
        project_name=project_name,
        experiment_name=experiment_name,
        run_name=run_name,
        config=config,
        artifacts=artifacts,
        metrics=metrics,
        extension=extension,
    )

    #
    mock_log_artifacts.assert_called_once_with(extension=extension, **artifacts)
    mock_log_metrics.assert_called_once_with(metrics)
    mock_log_params.assert_called_once_with(config)


@mock.patch("rambla.utils.mlflow.log_artifacts")
@mock.patch("rambla.utils.mlflow.mlflow.log_metrics")
@mock.patch("rambla.utils.mlflow.mlflow.log_params")
def test_mlflow_log_no_metrics(
    mock_log_params,
    mock_log_metrics,
    mock_log_artifacts,
):
    project_name = "mock_project_name"
    experiment_name = "mock_experiment_name"
    run_name = "mock_run_name"

    config = {"mock_config": "a"}
    artifacts = {"mock_artifacts": [1, 2, 3]}

    extension = "pkl"

    mlflow_log(
        project_name=project_name,
        experiment_name=experiment_name,
        run_name=run_name,
        config=config,
        artifacts=artifacts,
        extension=extension,
    )

    #
    mock_log_artifacts.assert_called_once_with(extension=extension, **artifacts)
    mock_log_metrics.assert_not_called()
    mock_log_params.assert_called_once_with(config)


@mock.patch("rambla.utils.mlflow.log_artifacts")
@mock.patch("rambla.utils.mlflow.mlflow.log_metrics")
@mock.patch("rambla.utils.mlflow.mlflow.log_params")
def test_mlflow_log_no_artifacts(
    mock_log_params,
    mock_log_metrics,
    mock_log_artifacts,
):
    project_name = "mock_project_name"
    experiment_name = "mock_experiment_name"
    run_name = "mock_run_name"

    config = {"mock_config": "a"}
    metrics = {"mock_metrics": 3.0}

    extension = "pkl"

    mlflow_log(
        project_name=project_name,
        experiment_name=experiment_name,
        run_name=run_name,
        config=config,
        metrics=metrics,
        extension=extension,
    )

    #
    mock_log_artifacts.assert_not_called()
    mock_log_metrics.assert_called_once_with(metrics)
    mock_log_params.assert_called_once_with(config)


@mock.patch("rambla.utils.mlflow.log_artifacts")
@mock.patch("rambla.utils.mlflow.mlflow.log_metrics")
@mock.patch("rambla.utils.mlflow.mlflow.log_params")
def test_mlflow_log_nothing_provided(
    mock_log_params,
    mock_log_metrics,
    mock_log_artifacts,
):
    project_name = "mock_project_name"
    experiment_name = "mock_experiment_name"
    run_name = "mock_run_name"

    mlflow_log(
        project_name=project_name,
        experiment_name=experiment_name,
        run_name=run_name,
    )

    #
    mock_log_artifacts.assert_not_called()
    mock_log_metrics.assert_not_called()
    mock_log_params.assert_not_called()
