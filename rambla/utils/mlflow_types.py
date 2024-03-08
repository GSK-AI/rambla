from typing import TypedDict


class MlflowRunInfoDict(TypedDict):
    artifact_uri: str
    end_time: int
    experiment_id: str
    lifecycle_stage: str
    run_id: str
    run_name: str
    run_uuid: str
    start_time: int
    status: str
    user_id: str
    experiment_name: str


class MlflowRunDataDict(TypedDict):
    metrics: dict
    params: dict
    tags: dict


class MlflowRunInputsDict(TypedDict):
    dataset_inputs: list


class MlflowRunDict(TypedDict):
    info: MlflowRunInfoDict
    data: MlflowRunDataDict
    inputs: MlflowRunInputsDict
