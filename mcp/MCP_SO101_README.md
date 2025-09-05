# SO101 Follower Arm MCP Server

このMCPサーバーは、LeRobotのSO101フォロワーアームを制御するためのModel Context Protocol (MCP)サーバーです。

## MCP仕様準拠

このサーバーは以下のMCP仕様に完全準拠しています：

- **JSON-RPC 2.0**: 標準化されたメッセージフォーマット
- **STDIO Transport**: プロセス間通信による効率的なデータ交換
- **Tools**: ロボット制御のための実行可能な機能
- **Resources**: リアルタイムデータへのアクセス
- **Prompts**: 使用ガイダンスとテンプレート
- **Error Handling**: 適切なエラーハンドリングと報告

## 機能

このMCPサーバーは、完全なModel Context Protocol仕様に準拠しており、以下の機能を提供します：

### 🔧 ツール (Tools)
- **ロボット接続**: SO101フォロワーアームへの接続と切断
- **グリッパー制御**: グリッパーの開閉制御（0-100の範囲）
- **関節制御**: 個別関節の位置制御
- **状態取得**: 現在のロボット状態と関節位置の取得

### 📊 リソース (Resources)
- **ロボット状態**: リアルタイムの関節位置と接続状態
- **ロボット設定**: 現在のロボット設定情報

### 💬 プロンプト (Prompts)
- **グリッパー制御テンプレート**: グリッパー操作のガイダンス
- **安全チェックテンプレート**: 移動前の安全確認手順

## 利用可能なツール

### 1. `connect_robot`
ロボットに接続します。

**パラメータ:**
- `port` (string): シリアルポート（例: '/dev/ttyUSB0' または 'COM3'）
- `calibrate` (boolean): 接続時にキャリブレーションを実行するか（デフォルト: true）

### 2. `disconnect_robot`
ロボットから切断します。

### 3. `get_robot_status`
現在のロボット状態と関節位置を取得します。

### 4. `control_gripper`
グリッパーを指定した位置に移動します。

**パラメータ:**
- `position` (number): グリッパー位置（0-100、0=開、100=閉）

### 5. `open_gripper`
グリッパーを完全に開きます。

### 6. `close_gripper`
グリッパーを完全に閉じます。

### 7. `move_joint`
指定した関節を目標位置に移動します。

**パラメータ:**
- `joint` (string): 関節名（shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll）
- `position` (number): 目標位置

### 8. `move_all_joints`
複数の関節を同時に移動します。

**パラメータ:**
- `shoulder_pan` (number): 肩パン位置
- `shoulder_lift` (number): 肩リフト位置
- `elbow_flex` (number): 肘屈曲位置
- `wrist_flex` (number): 手首屈曲位置
- `wrist_roll` (number): 手首回転位置
- `gripper` (number): グリッパー位置（0-100）

## 利用可能なリソース

### 1. `so101://robot/status`
現在のロボットの状態と関節位置をJSON形式で提供します。

**内容:**
- 接続状態
- 全関節の現在位置
- エラー情報（もしあれば）

### 2. `so101://robot/config`
現在のロボット設定情報をJSON形式で提供します。

**内容:**
- シリアルポート
- トルク無効化設定
- 最大相対目標値
- 角度単位設定

## 利用可能なプロンプト

### 1. `gripper_control`
グリッパー制御のためのガイダンスを提供します。

**引数:**
- `action` (required): 実行するアクション（open, close, position）

### 2. `safety_check`
ロボット操作前の安全チェック手順を提供します。

**引数:**
- `movement_type` (required): チェックする動作の種類

## セットアップ

### 1. 依存関係のインストール

```bash
# LeRobotの依存関係をインストール
cd /Users/sy/dev/lerobot
pip install -e .

# MCPサーバーの依存関係をインストール
pip install mcp
```

### 2. ロボットの接続

SO101フォロワーアームをUSBポートに接続し、適切なシリアルポートを確認してください。

### 3. MCPサーバーの実行

```bash
# 直接実行
python /Users/sy/dev/lerobot/mcp_so101_server.py

# または、MCPクライアントから設定ファイルを使用
# mcp_so101_config.jsonを参照
```

## 使用例

### 基本的なグリッパー制御

```python
# ロボットに接続
connect_robot(port="/dev/ttyUSB0")

# リソースでロボット状態を確認
read_resource(uri="so101://robot/status")

# プロンプトでグリッパー制御ガイダンスを取得
get_prompt(name="gripper_control", arguments={"action": "open"})

# グリッパーを開く
open_gripper()

# グリッパーを50%の位置に移動
control_gripper(position=50)

# グリッパーを閉じる
close_gripper()

# ロボットから切断
disconnect_robot()
```

### 関節制御

```python
# ロボットに接続
connect_robot(port="/dev/ttyUSB0")

# 個別関節の移動
move_joint(joint="shoulder_pan", position=0)
move_joint(joint="elbow_flex", position=45)

# 複数関節の同時移動
move_all_joints(
    shoulder_pan=0,
    shoulder_lift=30,
    elbow_flex=45,
    wrist_flex=0,
    wrist_roll=0,
    gripper=50
)

# 現在の状態を取得
status = get_robot_status()
print(status)
```

### 安全な操作手順

```python
# ロボットに接続
connect_robot(port="/dev/ttyUSB0")

# 安全チェックプロンプトを取得
get_prompt(name="safety_check", arguments={"movement_type": "gripper"})

# 現在の状態をリソースで確認
read_resource(uri="so101://robot/status")

# 安全を確認してからグリッパーを操作
control_gripper(position=30)

# 設定情報を確認
read_resource(uri="so101://robot/config")
```

## 注意事項

- ロボットを操作する前に、必ず安全な位置に配置してください
- 初回接続時はキャリブレーションが実行されます
- グリッパーの位置は0-100の範囲で指定してください（0=開、100=閉）
- ロボットの動作中は周囲の安全に注意してください

## トラブルシューティング

### 接続エラー
- シリアルポートが正しいか確認してください
- ロボットがUSBポートに正しく接続されているか確認してください
- 必要な権限があるか確認してください（Linux/macOSでは`sudo`が必要な場合があります）

### キャリブレーションエラー
- ロボットが安全な位置にあることを確認してください
- キャリブレーション手順に従ってください

### 動作エラー
- ロボットが接続されているか確認してください
- 関節の可動範囲内で位置を指定してください
