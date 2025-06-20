import json
import os
import pickle
from contextlib import contextmanager
from typing import IO, Any, BinaryIO, Callable, Dict, Iterator, TextIO, Union

import numpy as np
import torch
import yaml

__all__ = [
    "load",
    "save",
    "load_json",
    "save_json",
    "load_jsonl",
    "save_jsonl",
    "load_mat",
    "save_mat",
    "load_npy",
    "save_npy",
    "load_npz",
    "save_npz",
    "load_pt",
    "save_pt",
    "load_yaml",
    "save_yaml",
    "build_runname"
]


@contextmanager
def file_descriptor(f: Union[str, IO], mode: str = "r") -> Iterator[IO]:
    opened = False
    try:
        if isinstance(f, str):
            f = open(f, mode)
            opened = True
        yield f
    finally:
        if opened:
            f.close()


def load_json(f: Union[str, TextIO], **kwargs) -> Any:
    with file_descriptor(f, mode="r") as fd:
        return json.load(fd, **kwargs)


def save_json(f: Union[str, TextIO], obj: Any, **kwargs) -> None:
    with file_descriptor(f, mode="w") as fd:
        json.dump(obj, fd, **kwargs)


def load_jsonl(f: Union[str, TextIO], **kwargs) -> Any:
    with file_descriptor(f, mode="r") as fd:
        return [json.loads(datum, **kwargs) for datum in fd.readlines()]


def save_jsonl(f: Union[str, TextIO], obj: Any, **kwargs) -> None:
    with file_descriptor(f, mode="w") as fd:
        fd.write("\n".join(json.dumps(datum, **kwargs) for datum in obj))


def load_mat(f: Union[str, BinaryIO], **kwargs) -> Any:
    import scipy.io

    return scipy.io.loadmat(f, **kwargs)


def save_mat(f: Union[str, BinaryIO], obj: Any, **kwargs) -> None:
    import scipy.io

    scipy.io.savemat(f, obj, **kwargs)


def load_npy(f: Union[str, BinaryIO], **kwargs) -> Any:
    return np.load(f, **kwargs)


def save_npy(f: Union[str, BinaryIO], obj: Any, **kwargs) -> None:
    np.save(f, obj, **kwargs)


def load_npz(f: Union[str, BinaryIO], **kwargs) -> Any:
    return np.load(f, **kwargs)


def save_npz(f: Union[str, BinaryIO], obj: Any, **kwargs) -> None:
    np.savez(f, obj, **kwargs)


def load_pkl(f: Union[str, BinaryIO], **kwargs) -> Any:
    with file_descriptor(f, mode="rb") as fd:
        try:
            return pickle.load(fd, **kwargs)
        except UnicodeDecodeError:
            if "encoding" in kwargs:
                raise
            fd.seek(0)
            return pickle.load(fd, encoding="latin1", **kwargs)


def save_pkl(f: Union[str, BinaryIO], obj: Any, **kwargs) -> None:
    with file_descriptor(f, mode="wb") as fd:
        pickle.dump(obj, fd, **kwargs)


def load_pt(f: Union[str, BinaryIO], **kwargs) -> Any:
    return torch.load(f, **kwargs)


def save_pt(f: Union[str, BinaryIO], obj: Any, **kwargs) -> None:
    torch.save(obj, f, **kwargs)


def load_yaml(f: Union[str, TextIO]) -> Any:
    with file_descriptor(f, mode="r") as fd:
        return yaml.safe_load(fd)


def save_yaml(f: Union[str, TextIO], obj: Any, **kwargs) -> None:
    with file_descriptor(f, mode="w") as fd:
        yaml.safe_dump(obj, fd, **kwargs)


def load_txt(f: Union[str, TextIO]) -> Any:
    with file_descriptor(f, mode="r") as fd:
        return fd.read()


def save_txt(f: Union[str, TextIO], obj: Any, **kwargs) -> None:
    with file_descriptor(f, mode="w") as fd:
        fd.write(obj)


__io_registry: Dict[str, Dict[str, Callable]] = {
    ".txt": {"load": load_txt, "save": save_txt},
    ".json": {"load": load_json, "save": save_json},
    ".jsonl": {"load": load_jsonl, "save": save_jsonl},
    ".mat": {"load": load_mat, "save": save_mat},
    ".npy": {"load": load_npy, "save": save_npy},
    ".npz": {"load": load_npz, "save": save_npz},
    ".pkl": {"load": load_pkl, "save": save_pkl},
    ".pt": {"load": load_pt, "save": save_pt},
    ".pth": {"load": load_pt, "save": save_pt},
    ".pth.tar": {"load": load_pt, "save": save_pt},
    ".yaml": {"load": load_yaml, "save": save_yaml},
    ".yml": {"load": load_yaml, "save": save_yaml},
}


def load(fpath: str, **kwargs) -> Any:
    assert isinstance(fpath, str), type(fpath)

    for extension in sorted(__io_registry.keys(), key=len, reverse=True):
        if fpath.endswith(extension) and "load" in __io_registry[extension]:
            return __io_registry[extension]["load"](fpath, **kwargs)

    raise NotImplementedError(f'"{fpath}" cannot be loaded.')


def save(fpath: str, obj: Any, **kwargs) -> None:
    assert isinstance(fpath, str), type(fpath)
    os.makedirs(os.path.dirname(fpath), exist_ok=True)

    for extension in sorted(__io_registry.keys(), key=len, reverse=True):
        if fpath.endswith(extension) and "save" in __io_registry[extension]:
            __io_registry[extension]["save"](fpath, obj, **kwargs)
            return

    raise NotImplementedError(f'"{fpath}" cannot be saved.')

def rank0_print(*args, **kwargs):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)
    
    
def build_runname(args: Any, data_args:Any, spft_config:Any) -> str:
    
    #* Run Name Format:
    # {model_id}_{dataset}_b{per_device_train_batch_size}_ep{num_train_epochs}_lr{learning_rate}_{spft_setting}_{FFN-Sparsity}_{QKVO-Sparsity}_skip-{skip_mode}_start_{start_step}
    dataset_name = data_args.dataset.split(".")[0]
    run_name = f"{spft_config.model_id}_{dataset_name}_b{args.per_device_train_batch_size}_ep{args.num_train_epochs}_lr{args.learning_rate}_peft-{str(args.peft)}"
    
    #* Check SPFT Setting:
    run_name += f"_{spft_config.mode}"
    
    if "dense" in spft_config.mode:
        return run_name
    
    run_name += f"_{spft_config.ffn_sparsity}-{spft_config.qkvo_sparsity}"
    skip_mode = ""
    if spft_config.skip_output_tokens:
        skip_mode += "out-"
    if spft_config.skip_sink_tokens > 0:
        skip_mode += f"sink{spft_config.skip_sink_tokens}-"
    
    if skip_mode.endswith("-"):
        skip_mode = skip_mode[:-1]

    run_name += f"_skip-{skip_mode}"
    
    if spft_config.start_step > 0:
        run_name += f"_start_{spft_config.start_step}"
        
    return run_name
    
    