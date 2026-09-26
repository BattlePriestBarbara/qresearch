# Pre-Publish Audit — qresearch (公网发布前审查报告)

> **Audit marker:** `prepublish-audit:allow`
>
> 本文件是审计报告：它必须**引用**被判定为违规的绝对路径才能描述问题，因此按
> `scripts/prepublish_audit.py` 的约定声明豁免标记（该标记只豁免路径扫描，不豁免
> 密钥扫描；标记的出现会打印在每次审计输出中，绝不静默）。

| Field | Value |
|---|---|
| 审计对象 | 本仓库（qresearch，src-layout，`D:\Qlib`） |
| 审计时修订 | `fe67fef` (HEAD -> master)，工作区干净（审计过程为只读；本次新增/修改的文件见 6.3） |
| 审计日期 | 2026-09-18 |
| 审计执行 | Cline（自动化 + 人工交叉验证） |
| 跟踪文件 | 56 个，0.59 MB（审计基线） |
| git 对象库 | 125 objects，389.97 KiB |
| 工作区总大小 | 20.47 MB（其中 18.98 MB 为被忽略的 `.mypy_cache/`） |
| 自动化门禁 | `scripts/prepublish_audit.py`（本次新增，自带 17 例探测器自检） |

---

## 0. 结论摘要（TL;DR）

| # | 检查项 | 结论 | 阻断项 | 警告项 |
|---|---|---|---|---|
| 1 | 绝对路径排查（代码 / 配置文件） | **FAIL — 不可发布** | **50** | 33 |
| 2 | 大文件与数据隔离（`.gitignore` 严格生效） | **PASS**（可用，建议 4 项加固） | 0 | 10 |
| 3 | 敏感信息脱敏（API Key / 密码 / Token） | **PASS**（1 项中等风险：提交者身份） | 0 | 2 |
| — | **总计** | **NOT SAFE TO PUBLISH** | **50** | **45** |

复现命令（本次输出即由此产生，退出码 `1`）：

```powershell
python scripts/prepublish_audit.py --verbose     # 退出码 1 = 存在阻断项
```

**三句话结论：**

1. **数据与密钥是干净的**：没有任何 `.bin` / `.pth` / `.pkl` / 日志 / 数据集被跟踪，仓库体积
   0.59 MB（远低于"几十 MB"），`.gitignore` 与 pre-commit 的 `check-added-large-files` 双重生效；
   全历史扫描未发现任何 API Key、数据库口令或私人 Token。
2. **绝对路径是唯一的硬阻断**：50 处本地绝对路径落在**工具链会解析/执行**的文件里
   —— 其中 3 处是**功能性硬编码**（`src/qresearch/env.py`、`.pre-commit-config.yaml`、
   `configs/qlib_init.yaml`），其余是这些文件里的注释、docstring、以及由生成脚本写入的
   lock 文件头。任何人 clone 到别的机器后 `import qresearch` 会**直接抛
   `EnvironmentContractError` 终止进程**，pre-commit 门禁的 9 个 hook 也会全部
   "Executable not found"。
3. **另有一项非阻断但需要决策**：全部 7 个 commit 的作者身份是个人邮箱
   （已在下方脱敏为 `378***@qq.com`），历史一旦推送即公开；同时本地存在一个 IDE
   检查点引用 `refs/cline/checkpoints/...` 与 8 个不可达对象。

---

## 1. 审计方法

三道检查分别用两条独立路径交叉验证，避免"工具说干净就干净"：

| 检查 | 人工核对（git 原语） | 自动化门禁（本次新增） |
|---|---|---|
| 绝对路径 | `git grep -n -E "[A-Za-z]:[\\/]"` 限定**已跟踪文件** | `scan_absolute_paths()`：同样只扫描 `git ls-files` 的集合 |
| 数据隔离 | `git ls-files -i -c --exclude-standard`、`git check-ignore -v`、`git rev-list --objects --all`、`git count-objects -vH`、`git fsck --unreachable` | `scan_data_isolation()`：8 个子检查 |
| 敏感信息 | `git grep -i` 密钥/口令/Token 模式、`git config --local --list`、`git remote -v`、`git log --format='%ae'` | `scan_secrets()` + `scan_git_metadata()` |

自动化门禁自身的可信度证据（否则"扫描器说没问题"没有意义）：

```text
python scripts/prepublish_audit.py --selftest
→ selftest passed: 17 cases, detectors fire and stay silent as designed
```

17 例覆盖：Windows 盘符路径、UNC 路径、POSIX home、`$HOME` / `%USERPROFILE%`
（以上必须命中）；`https://` URL、`${QRESEARCH_DATA_ROOT}` 环境变量、本仓库 docstring 里的
LaTeX 数学（`\\not\\subseteq`）、环境变量间接取值、占位符 Token、noreply 邮箱
（以上必须**不**命中）。

**严重级别判定规则**（写入 `scripts/prepublish_audit.py` 文档字符串）：

* `BLOCKER` — 命中位置在**工具链加载的文件**（`src/`、`scripts/`、`tests/`、`configs/`
  以及根配置 `environment.yml`、`pyproject.toml`、`.pre-commit-config.yaml`、`.gitignore`、
  `.mypy.ini`、`.flake8`、`.pylintrc`、`requirements*.txt`）。注释与 docstring 也算阻断，
  因为"所有代码和配置文件严禁包含本地绝对路径"是无例外的要求。
* `WARNING` — 命中位置在文档（`README.md`、`PROJECT_SPEC.md`、`docs/`、`.gitkeep`）。
  文档可以描述机器布局，但公网读者会看到它；`--strict` 可将其升级为失败。


---

## 2. 检查一：绝对路径排查 — **FAIL（50 阻断 / 33 警告）**

### 2.1 阻断项（按文件归并，共 50 处）

| 文件 | 处数 | 行号 | 性质与影响 |
|---|---|---|---|
| `.pre-commit-config.yaml` | 13 | 21,23,25,28,34,70,78,91,98,104,119,149,158 | **功能性**：9 个 hook 的 `entry: D:/Anaconda3/python.exe -m <tool>`。换机器后门禁全部失败；其余 4 处是说明注释 |
| `src/qresearch/env.py` | 6 | 11,60,67,68,498,503 | **功能性**：`EXPECTED_INTERPRETER`(60)、`DEFAULT_PROVIDER_URI`(67)、`DEFAULT_CACHE_ROOT`(68) 直接写死 `D:\...`；导入时校验与默认数据根都依赖它 |
| `configs/qlib_init.yaml` | 8 | 11,12,16,25,26,33,40,42 | **功能性**：`provider_uri: "D:/qlib_data/cn_data"`(16)、`uri: "file:D:/Qlib/artifacts/mlruns"`(33)；25/26 是注释中的示例缓存路径 |
| `scripts/lock_requirements.py` | 5 | 42,43,131,286,309 | **功能性**：docstring 用法、报错提示与 `_header()` 内的字符串**会重新写回** lock 文件（即使手工清理 lock，下次 `--hashes` 又会被污染） |
| `requirements.lock.txt` | 3 | 5,9,18 | 生成物头部的 "Regenerate / Interpreter / Install" 三行 |
| `requirements.lock.hashes.txt` | 2 | 5,9 | 同上（第 9 行 `# Interpreter: D:\Anaconda3\python.exe` 由 `sys.executable` 动态写入，见 2.3） |
| `requirements.lock.pip.txt` | 2 | 5,9 | 同上 |
| `src/qresearch/__init__.py` | 2 | 3,10 | 包级 docstring 宣称 "project root `D:\Qlib`"、解释器为 `D:\Anaconda3\python.exe` |
| `tests/test_env_contract.py` | 3 | 94,104,110 | 负例夹具用了 `D:\msys64\mingw64\bin\python.exe`、`C:\Python310\python.exe`（字符串本身无害，但泄露开发机布局，且断言写死了输出文本） |
| `.gitignore` | 1 | 51 | 注释中记录市场数据绝对路径 |
| `.mypy.ini` | 1 | 8 | 注释中的运行命令 |
| `environment.yml` | 1 | 62 | 注释中的安装命令 |
| `tests/conftest.py` | 1 | 18 | `repo_root` docstring 写死 `D:\Qlib` |
| `tests/test_provider_readonly.py` | 1 | 4 | docstring 写死 `D:\qlib_data\cn_data` |
| `tests/test_repo_structure.py` | 1 | 81 | **该测试把绝对路径当成契约**：`assert "D:/Anaconda3/python.exe -m mypy" in config` |

> 行号对应审计修订 `fe67fef`（已按本次新增 `pre-publish` hook 后的位移校正）。行号会随文件
> 演化漂移，随时可用 `python scripts/prepublish_audit.py --verbose` 取到当前值。

**最严重的一处**（唯一会让"换机器即崩"的地方）：

```python
# src/qresearch/env.py:60,67,68
EXPECTED_INTERPRETER: Final[Path] = Path(r"D:\Anaconda3\python.exe")
DEFAULT_PROVIDER_URI: Final[Path] = Path(r"D:\qlib_data\cn_data")
DEFAULT_CACHE_ROOT:  Final[Path] = Path(r"D:\Qlib\artifacts\qlib_cache")
```

`verify_environment(require_pinned_interpreter=True)` 是默认参数，且 `qresearch/__init__.py` 第 46 行
在**导入期**调用它，所以上述路径在别的机器上必然导致 `EnvironmentContractError` 并终止进程。

### 2.2 警告项（文档类，33 处，不阻断但建议一并清理）

| 文件 | 处数 | 说明 |
|---|---|---|
| `README.md` | 17 | 行 13,14,16,29,72,74,75,78,81,108,113,132,145,170,172,173,219：项目根、解释器、数据根、安装命令、目录树、`icacls` 授权命令、门禁命令表、数据快照输出 |
| `PROJECT_SPEC.md` | 14 | 行 12,14,15,16,872,884,891,965,983,1145,1154,1285,1501,1502：规范表头、目录树、示例 `qlib_init`、验收标准 |
| `data/.gitkeep` | 1 | 行 2 |
| `docs/adr/ADR-001-mypy-runs-in-pinned-environment.md` | 1 | 行 19 |

> `PROJECT_SPEC.md` 是**规范性文件**（README 声明"与之矛盾即缺陷"）。因此第 2.4 节的修复
> 必须同时提交一份 ADR 并修订规范条文，否则修复动作本身违反仓库治理约定。

### 2.3 为什么不能简单 `sed` 替换（4 个耦合点）

1. **测试把字面量当契约**：`tests/test_repo_structure.py:81` 断言配置里必须出现
   `D:/Anaconda3/python.exe -m mypy`。改配置必须同改该断言，否则门禁红。
2. **lock 生成器会写回**：`scripts/lock_requirements.py` 的 `_header()`（行 286）注入字面量，
   而第 290 行 `f"# Interpreter: {sys.executable}"` 写的是**运行机器上的真实路径**。
   换句话说：只要本机跑一次 `--hashes`，绝对路径就重新出现在 3 个 lock 文件里。
   要么只保留 `python scripts/lock_requirements.py` 这种可移植写法并去掉 Interpreter 行，
   要么把 `sys.executable` 归一化（例如 `Path(sys.executable).name`）。
3. **契约与环境创建流程自相矛盾**：`environment.yml` 定义的是具名环境 `qresearch`，
   `conda env create` 后的解释器应为 `<conda_root>\envs\qresearch\python.exe`，
   而契约写的是 **base** 的 `D:\Anaconda3\python.exe`。这说明"钉死路径"在跨机器/跨安装方式上
   本来就不可靠——正确做法是把解释器变成**声明式输入**而不是常量。
4. **`test_no_bypass_environment_variable_exists`**（`tests/test_env_contract.py:114`）禁止任何
   "跳过校验"的环境变量。任何引入环境变量覆盖的修复都必须论证它**不放宽**严格性
   （覆盖"解释器*位置*"≠ 关闭"必须匹配钉死版本*集合*"的校验），并同步该测试。

### 2.4 修复方案（按顺序执行，每步都可单独验证）

> 前置：`PROJECT_SPEC.md` 是规范文件，本修复改变了接口常量与配置解析方式 ⇒ 先落一份
> ADR（`docs/adr/ADR-005-*.md`）并修订 `PROJECT_SPEC.md` 3.1 / 3.6 的相关条文。

**Step 1 — 把"位置"变成声明式输入（核心）**
新增 `configs/environment.yaml`（或复用 `pyproject.toml` 的 `[tool.qresearch]`）：

```yaml
environment:
  python: "3.12.7"                        # 版本契约保持不变
  interpreter: "%QRESEARCH_PYTHON%"       # 未设置时回退 sys.executable
  provider_uri: "%QRESEARCH_DATA_ROOT%"   # 未设置时必须显式报错，不猜路径
  cache_root: "artifacts/qlib_cache"      # 相对仓库根，解析时再转绝对
```

`src/qresearch/env.py` 改为：读取该文件 → 环境变量覆盖 → 未设置时 `sys.executable`；
**版本集合、禁令标记（msys64/mingw64/WindowsApps）与"无绕过变量"三条严格性全部保留**，
只有"期望的解释器路径"从常量变成配置。`--provider-uri/--cache-root` 命令行参数已存在，无需改动。

**Step 2 — `.pre-commit-config.yaml` 去掉 9 处绝对路径**
推荐引入 `scripts/hook_runner.py`，hook 写 `entry: python scripts/hook_runner.py -m black`：
runner 解析 Step 1 的配置选择解释器并在其不可用时**报错退出**，从而保留 ADR-001 的意图
（mypy 必须跑在钉死环境里），同时消除机器路径。`entry` 仍需使用正斜杠（既有注释已说明
pre-commit 的 POSIX shlex 会吞掉反斜杠）。

**Step 3 — `configs/qlib_init.yaml`**
`provider_uri: "${QRESEARCH_DATA_ROOT}/cn_data"`、`uri: "file:artifacts/mlruns"`。
该文件本就是 `build_qlib_init_kwargs()` 的"声明式镜像"，绝对路径以函数返回值为准。

**Step 4 — 测试与 docstring**
`tests/test_repo_structure.py:81` 改为断言"配置中不含盘符路径，且解释器来自配置解析"；
`tests/test_env_contract.py:110` 的期望文本用 `repr(Path(...))` 组合生成而非写死；
`tests/conftest.py:18`、`tests/test_provider_readonly.py:4`、`src/qresearch/__init__.py` 的
docstring 改为 `%QRESEARCH_DATA_ROOT%` 之类的占位描述。

**Step 5 — `scripts/lock_requirements.py`**
删除 5 处字面量；`_header()` 第 286 行改为 `python scripts/lock_requirements.py [--hashes]`；
第 290 行 `# Interpreter:` 改为 `Path(sys.executable).name`（或删掉该行），否则重新生成
lock 时绝对路径必然再现。

**Step 6 — 文档占位化**
`README.md` / `PROJECT_SPEC.md` 用 `%QRESEARCH_DATA_ROOT%`、`<repo>\artifacts`、
`python -m <tool>` 取代 `D:\...`（合计 31 处警告）。`icacls` 示例改为
`icacls "%QRESEARCH_DATA_ROOT%\cn_data" /deny ...`。

**Step 7 — 验收（必须全绿）**

```powershell
python scripts/prepublish_audit.py              # 期望 exit 0
python scripts/prepublish_audit.py --strict     # 期望 exit 0（文档也干净）
python scripts/prepublish_audit.py --selftest   # 探测器自检 17/17
python -m pytest -q -m "not slow"               # 门禁测试
pre-commit run --all-files
```

**关于历史**：本次审计**不要求**改写历史——历史里没有密钥、没有数据、没有大文件，
只有这些同一批文件的旧版本（含文档中的机器路径）。若要连"旧版本里的机器路径"也不公开，
唯一办法是重写历史（`git filter-repo`）或用一次干净的初始提交另起仓库；这是一项
独立的、破坏性的决策，需要你确认后再执行。

---

## 3. 检查二：大文件与数据隔离 — **PASS（0 阻断 / 10 警告）**

### 3.1 已证明通过的项

| 证据 | 命令 | 结果 |
|---|---|---|
| `.gitignore` 严格生效 | `git ls-files -i -c --exclude-standard` | **空**：没有任何被跟踪文件同时命中忽略规则，即不存在"该忽略却被提交"的文件 |
| 无数据/权重/日志被跟踪 | `git ls-files` 过滤 33 类后缀（`.bin .pth .pt .ckpt .pkl .pickle .joblib .h5 .hdf5 .onnx .npz .npy .safetensors .model .weights .log .parquet .db .sqlite3 .zip .gz .tar .pem .key .pfx .p12` …） | **空**；跟踪集合只有 `.py`(31) `.md`(6) `.txt`(5) `.gitkeep`(5) `.yaml`(2) 及 `.typed .toml .yml .ini .gitignore .flake8 .pylintrc` |
| 无二进制内容混入 | 逐文件取前 8 字节检测 NUL | 无 |
| 无符号链接 | `git ls-files -s` 无 `120000` 模式 | OK |
| 仓库对象体积 | `git count-objects -vH` | 125 objects / **389.97 KiB**（loose，无 pack） |
| 历史最大 blob | `git rev-list --objects --all` + `git cat-file -s`（阈值 100 KB） | 仅 `requirements.lock.hashes.txt` 177 KB（必需）、`PROJECT_SPEC.md` 99 KB |
| 跟踪内容体积 | 56 个跟踪文件求和 | **0.59 MB** |
| 工作区体积 | 递归统计（含被忽略项） | 20.47 MB，其中 `.mypy_cache/` 占 18.98 MB（已忽略，不入库） |
| 提交时大文件防线 | pre-commit `check-added-large-files --maxkb=2048` | 单文件 >2 MB 无法提交（第二道防线） |
| 忽略规则抽样有效性 | `git check-ignore -v` | IGNORED：`data/cn_data/features/**/*.bin`、`artifacts/mlruns/0/meta.yaml`、`artifacts/models/model.pth`、`model.pkl`、`logs/train.log`、`.env`、`credentials.json`、`secrets.yaml` |

结论：**Qlib 二进制数据、模型权重、庞大日志都没有、也不可能被推送到 GitHub**；
仓库体积约 0.6 MB，远小于"几十 MB"的要求。

### 3.2 缺口与加固建议（4 项，均不阻断，属"把工作区也变安全"）

1. **`cn_data/` 位于仓库根时未被忽略**（真实风险）。现有规则 `data/**` 只保护 `data/` 目录，
   而 Qlib 存储通常整体复制（`cp -r D:\qlib_data\cn_data .`）⇒ `cn_data/instruments/all.txt`
   与 `cn_data/calendars/day.txt` **是可提交的**（`.bin` 被忽略，`.txt` 不会）。
   建议：`.gitignore` 增加**根锚定**规则 `/cn_data/`、`/calendars/`、`/instruments/`
   （必须根锚定，否则会误伤 `src/qresearch/features/` 这类包目录）。
2. **派生数据缺少类型级防线**：`predictions.parquet`、`results.csv`、`*.feather`、`*.arrow`、
   `*.db`、`*.sqlite3`、`*.zarr/` 均未被忽略。建议直接加入 `.gitignore`，或在规范中明确
   "派生数据只允许写入 `artifacts/`（已被 `artifacts/**` 全量忽略）"。
3. **本机覆盖配置未忽略**：`configs/local.yaml`。建议 `configs/local*.yaml`，
   避免有人把本机路径/端口写进被跟踪的配置。
4. **本地 git 杂物**：8 个不可达对象 + 1 个 IDE 检查点引用
   `refs/cline/checkpoints/1789448214262_68j1v/2`（内容为 40 个源码文件的快照，
   **不含** 数据/权重/日志，本次已核实）。普通 `git push` 不会推送非分支引用，
   但 `git push --mirror`、`git bundle`、或直接打包整个目录都会带上。
   发布前建议：`git gc --prune=now`；如需彻底干净再 `git update-ref -d <该引用>`。

---

## 4. 检查三：敏感信息脱敏 — **PASS（0 阻断 / 2 警告）**

### 4.1 已证明干净的项

| 证据 | 命令 | 结果 |
|---|---|---|
| API Key / Token / 口令模式 | `git grep -i -E "(api[_-]?key\|apikey\|secret\|password\|passwd\|token\|bearer\|credential\|private[_-]?key\|BEGIN .*PRIVATE KEY\|AKIA[0-9A-Z]{16}\|ghp_[A-Za-z0-9]{20,}\|sk-[A-Za-z0-9]{16,})"` | 仅 6 处**无害**命中：`.gitignore` 的注释与 `credentials.json`/`secrets.yaml` 规则名；`.pre-commit-config.yaml` 的 `detect-private-key` hook id；`asttokens==2.0.5`（包名含 "stoken"，误报） |
| 邮箱（文件内 PII） | `git grep -E "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"` | **空**：跟踪文件中没有任何邮箱地址 |
| 凭据型 URL / 内网地址 | `git grep -i -E "(localhost\|127\.0\.0\.1\|http://\|mongodb://\|postgres://\|mysql://\|redis://\|ftp://\|amqp://)"` | **空** |
| 私钥/凭据文件 | `git ls-files` 后缀过滤 `.pem .key .pfx .p12 .jks .keystore .env .env.*` | **空**；`.gitignore` 已覆盖（含 `!.env.example` 白名单，但该文件当前不存在） |
| 本地 git 配置 | `git config --local --list` | 仅 `core.*`（无 `url.<...>.insteadOf`、无 token） |
| 远端 | `git remote -v` | **空**（尚未配置远端，因此不存在"URL 里嵌 token"的经典泄露） |
| 本地历史 | `git rev-list --objects --all` 全量 blob 扫描 | 无密钥、无数据、无大文件（见 3.1） |

### 4.2 需要你决策的一项（中等风险）：提交者身份

7 个 commit（含 Cline 的 2 个 IDE 检查点提交）的作者/提交者均为个人邮箱
`378***@qq.com`（此处已脱敏；原值见 `git log --format='%ae'`）。邮箱会随历史永久公开，
且**无法通过后续提交删除**。三个选项：

| 选项 | 操作 | 代价 |
|---|---|---|
| A. 接受 | 不做处理 | 个人邮箱公开 |
| B. 改成 noreply | `git filter-repo --email-callback ...`（或重建仓库） | 历史被重写，需强推；本地一切 clone 失效 |
| C. 单次提交发布 | 用一次干净的初始提交另起仓库 | 丢失真实开发历史 |

另需注意：`refs/cline/checkpoints/1789448214262_68j1v/2` 是 IDE 会话状态（源码快照），
属本地开发痕迹；不影响 `git push`，但会随目录打包外泄（见 3.2 第 4 条）。

### 4.3 建议

* 发布前再跑一次**全历史**扫描（本报告只覆盖可静态识别的模式）：
  `gitleaks detect --no-banner` 或 `trufflehog git file://. --only-verified`。
* 把本审计脚本接入 pre-push / CI（`PROJECT_SPEC.md` 的 `INF-12` CI 质量门禁正是它的合适归宿）。
* GitHub 侧开启 **secret scanning + push protection**：即使本次代码干净，未来一次误提交也会被拦下。

---

## 5. 发布前 Checklist（可直接复制的命令序列）

```powershell
# 0) 本次审计的自动化门禁（期望：修完 Step 1..6 后 exit 0）
python scripts/prepublish_audit.py                # 阻断项 -> 非 0
python scripts/prepublish_audit.py --strict       # 文档告警也算失败
python scripts/prepublish_audit.py --selftest      # 探测器自检

# 1) 仓库自身门禁（fast + slow）
pre-commit run --all-files
pre-commit run --all-files --hook-stage manual

# 2) 体积 / 杂物整理
git count-objects -vH
git gc --prune=now                       # 清掉 8 个不可达对象

# 3) （可选）删除 IDE 检查点引用
git update-ref -d refs/cline/checkpoints/1789448214262_68j1v/2

# 4) 远端与推送（当前无远端，注意不要用 --mirror，否则会带上非分支引用）
git remote add origin <your-repository-url>
git push -u origin master
```

**不要做**：`git push --mirror`（会推送 `refs/cline/**`）、`git push --all --prune` 之外的
`--force` 组合，以及在未完成 Step 1..6 前推送。

---

## 6. 复现方式与本次变更清单

### 6.1 一键复现

```powershell
python scripts/prepublish_audit.py --verbose
```

`--verbose` 会逐条列出全部 50 个阻断项与 45 个警告项，即本报告第 2–4 节的原始数据来源。

### 6.2 审计脚本的判定规则

| 退出码 | 含义 |
|---|---|
| `0` | 干净 |
| `1`（bit） | 机器加载文件中存在绝对路径 |
| `2`（bit） | 数据隔离 / 仓库体积违规 |
| `4`（bit） | 密钥、凭据或身份发现 |
| `8` | 审计自身无法完成（例如 git 不可用） |

位或组合（如 `3` = 路径 + 数据）。`--strict` 把 `WARNING` 计为失败。
文件可通过 `prepublish-audit:allow` 标记豁免**路径**扫描（例如本报告），
但**密钥/凭据永不豁免**，且豁免文件名每次都会打印。

### 6.3 本次新增/改动的文件

审计过程的 git 查询全部只读（不写仓库、不写索引）；除下表外，第 2 节列出的文件
**一律未被改动**。

| 文件 | 性质 | 说明 |
|---|---|---|
| `docs/audits/PRE-PUBLISH-AUDIT.md` | 新增 | 本报告 |
| `scripts/prepublish_audit.py` | 新增 | 自动化门禁（只读 git 查询，不写仓库/索引） |
| `tests/test_prepublish_audit.py` | 新增 | 探测器单元测试 + `slow` 标记的"发布就绪"端到端测试（默认门禁用 `-m "not slow"` 排除了它，因此修复期间的红色状态不会污染快速门禁） |
| `.pre-commit-config.yaml` | 修改 | 仅新增 `pre-publish audit` hook（`stages: [pre-push]`，`entry: python scripts/prepublish_audit.py`）；**未**改动既有 hook 或本次报告要求清理的 entry |

既有代码：**未改动**。第 2.4 节的修复涉及接口常量与规范条文，按仓库治理约定需要
先写 ADR 并修订 `PROJECT_SPEC.md`，因此留给下一批提交，避免把"审查"与"改造"混在一起。
新 hook 在 `pre-push` 阶段才会执行，所以它**不会**影响日常提交，只会在真正推送前拦下违规；
在 Step 1..6 完成前它就是红的——这正是它的用途。

---

## 附录 A：自动化门禁输出摘要（本次实跑）

```text
qresearch pre-publish audit
  repository  : D:\Qlib
  revision    : fe67fef  (HEAD -> master)
  tracked     : 56 file(s)
  allow-listed: none

[1/3 absolute paths in machine-loaded files] FAIL - 50 blocker(s), 33 warning(s)
[2/3 data isolation and repository size]     REVIEW - 0 blocker(s), 10 warning(s)
[3/3 secrets, credentials and identity]      REVIEW - 0 blocker(s), 2 warning(s)

VERDICT: NOT SAFE TO PUBLISH - 50 blocker(s) (exit code 1)
```

```text
python scripts/prepublish_audit.py --selftest
selftest passed: 17 cases, detectors fire and stay silent as designed
```

> 上面两段取自**审计基线**（新增本次文件之前：`tracked: 56`、`allow-listed: none`）。
> 把新增文件纳入索引后复跑，结果为 `tracked: 59`、
> `allow-listed: docs/audits/PRE-PUBLISH-AUDIT.md, scripts/prepublish_audit.py`，
> 三节计数与结论完全一致（50 阻断 / 33 警告、0 / 10、0 / 2）——说明新增文件没有引入新问题，
> 而记录了大量违规路径的本报告也没有被误报（豁免机制按设计工作）。

## 附录 B：与"发布前三条要求"的逐条对照

| 要求 | 判定 | 证据 | 残留风险 / 待办 |
|---|---|---|---|
| 1. 严禁本地绝对路径（代码与配置） | **不满足** | 50 个阻断项（第 2.1 节）；其中 `src/qresearch/env.py`、`.pre-commit-config.yaml`、`configs/qlib_init.yaml` 为功能性硬编码 | 按第 2.4 节 Step 1–7 修复；修复需 ADR + 规范修订 |
| 2. `.gitignore` 严格生效、数据/权重/日志不入库、体积几十 MB 内 | **满足** | `.gitignore` 命中检查为空；无任何数据/权重/日志被跟踪；对象库 389.97 KiB、跟踪内容 0.59 MB；历史最大 blob 177 KB | 4 项加固建议（`cn_data/` 根锚定、派生数据后缀、`configs/local*.yaml`、`git gc`） |
| 3. 无硬编码 API Key / 数据库口令 / 私人 Token | **满足** | 全量模式扫描无非预期命中；无 `.env`/`.pem`/`credentials.json` 等；`.git/config` 无 token；无远端 URL | 决策项：提交者个人邮箱随历史公开；建议全历史 `gitleaks`/`trufflehog` 复扫 + 远端 secret scanning |

---

## 附录 C：修复复核（2026-09-26，随 `refactor(config)` 提交落库）

> 第 0–6 节是**审计基线原文，一字未改**——审计报告若被"顺手修正"就失去可追溯性。
> 本附录记录第 2.4 节 Step 1..6 落地后的复跑数值，以及仍然剩余的 7 条警告的归属。

| Field | 基线（第 0 节） | 复核（本附录） |
|---|---|---|
| 审计时修订 | `fe67fef` | `f3a19d4`（复核时 HEAD；修复尚未提交，见 C.4） |
| 跟踪文件 | 56 | 58 |
| 检查一 绝对路径 | **FAIL** — 50 阻断 / 33 警告 | **PASS** — 0 阻断 / 0 警告 |
| 检查二 数据隔离 | REVIEW — 0 / 10 | REVIEW — 0 / 1 |
| 检查三 密钥身份 | REVIEW — 0 / 2 | REVIEW — 0 / 6 |
| 总计 | **NOT SAFE TO PUBLISH** — 50 / 45 | **SAFE TO PUBLISH** — 0 / 7 |
| 退出码 | `1` | `0` |

### C.1 复核输出（`python scripts/prepublish_audit.py`）

```text
qresearch pre-publish audit
  repository  : D:\Qlib
  revision    : f3a19d4  (HEAD -> master)
  tracked     : 58 file(s)
  allow-listed: none

[1/3 absolute paths in machine-loaded files] PASS - 0 blocker(s), 0 warning(s)

[2/3 data isolation and repository size] REVIEW - 0 blocker(s), 1 warning(s)
  WARNING git objects                                          17 unreachable object(s); `git gc --prune=now` before mirroring

[3/3 secrets, credentials and identity] REVIEW - 0 blocker(s), 6 warning(s)
  WARNING git history                                          20 commit(s) by 378***@qq.com; published with the history
  WARNING refs/cline/checkpoints/1789448214262_68j1v/2         local-state ref; a mirror push or an archive carries it
  WARNING refs/cline/checkpoints/1789743861866_wlkdc/2         local-state ref; a mirror push or an archive carries it
  WARNING refs/cline/checkpoints/1790419144991_qvyfg/1         local-state ref; a mirror push or an archive carries it
  WARNING refs/cline/checkpoints/1790419144991_qvyfg/2         local-state ref; a mirror push or an archive carries it
  WARNING refs/cline/checkpoints/1790419144991_qvyfg/3         local-state ref; a mirror push or an archive carries it

VERDICT: SAFE TO PUBLISH - 7 warning(s), 0 blockers
```

`--selftest` 仍为 17/17（探测器未被修复过程削弱）；仓库自身门禁同步全绿：
`pre-commit run --all-files`、`--hook-stage manual`（pylint、pytest）、`--hook-stage pre-push`
（含 `pre-publish-audit` hook）三档均 `Passed`，`pytest -q -m "not slow"` 为
`260 passed, 1 deselected`（基线 173 passed）。

### C.2 Step 1..6 的落点

| Step | 落地方式 | 关键文件 |
|---|---|---|
| 1 | 位置改为**声明式解析**（不走 `configs/environment.yaml`，改由 `paths.py` 统一解析环境变量 + 文档化回退，理由见 ADR-005"被否方案"） | `src/qresearch/config/paths.py`、`config/errors.py`、`env.py` |
| 2 | 9 个 hook 的 entry 统一改为 `python scripts/hook_runner.py -m <tool>`，解释器在运行时解析 | `.pre-commit-config.yaml`、`scripts/hook_runner.py` |
| 3 | `qlib_init.yaml` 模板化（`${QLIB_DATA_DIR}`、`file:artifacts/mlruns`） | `configs/qlib_init.yaml` |
| 4 | 测试与 docstring 去字面量（断言改为"不得含盘符路径"） | `tests/test_repo_structure.py`、`tests/test_env_contract.py`、`tests/test_provider_readonly.py`、`tests/conftest.py`、`src/qresearch/__init__.py` |
| 5 | lock 头改为 `python scripts/lock_requirements.py`，解释器行改为 `Path(sys.executable).name` | `scripts/lock_requirements.py`、三份 `requirements.lock*.txt` |
| 6 | 文档占位化（`%QLIB_DATA_DIR%`、`<repo>/artifacts`、`python -m <tool>`） | `README.md`、`PROJECT_SPEC.md`（v1.1.0）、`docs/adr/ADR-001-*.md` |
| 7 | 验收命令全绿（C.1）；新增的 `tests/test_config_paths.py` 复用审计脚本自己的规则做**反向断言**，防止回归 | 见 C.3 |

Step 1 的偏差（配置文件名、`dataclasses` 而非 pydantic、回退而非硬报错）逐条记录在
`docs/adr/ADR-005-path-and-config-management.md`，`PROJECT_SPEC.md` 升到 v1.1.0 并新增 §3.1.1。

### C.3 剩余 7 条警告的归属（全部需要"人的动作"，不是代码缺陷）

| 警告 | 数量 | 归属与处置 |
|---|---|---|
| 不可达 git 对象 | 1 | 第 5 节 Step 2 的 `git gc --prune=now`。**未自动执行**：它会不可逆地丢弃 IDE 检查点所用对象，属数据处置决策 |
| 提交者邮箱随历史公开 | 1 | 第 4.2 节的中等风险决策（改写历史 / 接受公开 / 另起干净仓库） |
| `refs/cline/checkpoints/**` 本地引用 | 5 | 第 5 节 Step 3 的 `git update-ref -d`；引用由 IDE 生成，删除会同时移除其回滚能力，交由使用者决定 |

因此 `python scripts/prepublish_audit.py --strict` 仍为红（`--strict` 把警告计为失败），
`tests/test_prepublish_audit.py::test_repository_is_publish_ready`（标记 `slow`）在完成
上述三项人为动作前**保持红色是设计意图**：默认门禁 `-m "not slow"` 不受影响，而任何把该测试
改绿的做法都等于放宽判定标准。

数据隔离一节的 9 条加固建议（`cn_data/`、`calendars/`、派生数据后缀、`configs/local*.yaml`）
已按建议写入 `.gitignore`，使该节从 10 条警告降到 1 条；新增规则经
`git ls-files -i -c --exclude-standard` 验证**没有**误伤任何已跟踪文件。

### C.4 复核时的工作区状态

复核在修复**尚未提交**时执行，因此脚本打印的 `revision` 仍是父修订 `f3a19d4`
（`tracked: 58`，不含本附录所在报告与 `scripts/prepublish_audit.py` 等新文件尚未入索引的部分）。
把新增文件纳入索引后复跑，预期 `tracked` 增加、`allow-listed` 列出报告与脚本自身，
而三节计数与本附录一致。提交后请按第 5 节命令序列复跑一次作为发布前的最终证据。

### C.5 纳入索引后的最终复跑（最有说服力的一次）

C.1 的 `tracked: 58` 只覆盖**旧**文件——新文件此前是未跟踪状态，因此**不在扫描范围**。
把本批 41 个文件（20 个新增 + 21 个修改）加入索引后复跑，扫描面才对全部 78 个跟踪文件生效：

```text
qresearch pre-publish audit
  repository  : D:\Qlib
  revision    : f3a19d4  (HEAD -> master)
  tracked     : 78 file(s)
  allow-listed: docs/audits/PRE-PUBLISH-AUDIT.md, scripts/prepublish_audit.py

[1/3 absolute paths in machine-loaded files] PASS - 0 blocker(s), 0 warning(s)

[2/3 data isolation and repository size] REVIEW - 0 blocker(s), 1 warning(s)
  WARNING git objects                                          19 unreachable object(s); `git gc --prune=now` before mirroring

[3/3 secrets, credentials and identity] REVIEW - 0 blocker(s), 6 warning(s)
  WARNING git history                                          20 commit(s) by 378***@qq.com; published with the history
  WARNING refs/cline/checkpoints/1789448214262_68j1v/2         local-state ref; a mirror push or an archive carries it
  WARNING refs/cline/checkpoints/1789743861866_wlkdc/2         local-state ref; a mirror push or an archive carries it
  WARNING refs/cline/checkpoints/1790419144991_qvyfg/1         local-state ref; a mirror push or an archive carries it
  WARNING refs/cline/checkpoints/1790419144991_qvyfg/2         local-state ref; a mirror push or an archive carries it
  WARNING refs/cline/checkpoints/1790419144991_qvyfg/3         local-state ref; a mirror push or an archive carries it

VERDICT: SAFE TO PUBLISH - 7 warning(s), 0 blockers
```

这一次复跑**抓到了一个真实缺陷**，值得记下来：`src/qresearch/config/loader.py` 的
`as_posix()` docstring 里为解释转义陷阱而写的 `"C:\\Users"` 字面量，被
`windows-drive-path` 规则判为阻断项（"说明路径"仍然是路径）。它此前不可见，只因为这些新文件
尚未跟踪——**未被跟踪的文件不在审计范围内**，这是审计方法的固有边界，而不是脚本缺陷。
处置：把该 docstring 改为不含盘符的措辞（不豁免源码文件——给源码加豁免标记等于关掉这条规则）。

同一根因还暴露了 `tests/test_config_paths.py` 的一个漏洞：它用审计脚本的
`PATH_PATTERNS` 做反向断言，却没有复刻审计的豁免机制，于是会误报审计脚本自身的探测规则。
现已改为**镜像**该机制（同名常量 `AUDIT.ALLOW_MARKER`），并把"扫描根目录内被豁免的文件集合"
钉死为 `EXPECTED_EXEMPT_IN_SCANNED_ROOTS`，使任何生产代码都无法悄悄获得豁免。




> 本报告自身的所有绝对路径都写在文档里（属 `WARNING` 级别，且已声明 `prepublish-audit:allow`）；
> 修复完成后请用 `--strict` 复跑，让文档也一并干净。
