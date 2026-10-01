"""``python -m fundeval``：与命令行 ``fundeval`` 相同，便于未安装、以 ``PYTHONPATH=src`` 运行源码时使用。"""

from fundeval.cli import main

raise SystemExit(main())
