"""版本号的唯一来源：pyproject.toml 通过 ``[tool.setuptools.dynamic]`` 读取这里的 ``__version__``。

不依赖已安装包的元数据（importlib.metadata），因此以 ``PYTHONPATH=src`` 直接运行源码时
``fundeval.__version__`` 与 ``fundeval --version`` 同样可用。
"""

__version__ = "0.2.0"
