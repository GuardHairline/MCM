# 邮件通知配置指南

## 📧 功能说明

实验完成后，系统会自动发送邮件通知，包含：
- ✅ 成功完成的实验列表
- ❌ 失败的实验列表及错误信息
- ⏱️ 时间统计和总耗时
- 📊 成功率统计

---

## 🚀 快速开始

### 方法1: 使用环境变量（推荐）

```bash
# 设置SMTP密码（使用163邮箱授权码）
export SMTP_PASSWORD="your_authorization_code"

# 运行实验（指定收件邮箱）
bash scripts/configs/autodl_config/run_autodl_experiments.sh \
    --email your_email@example.com
```

### 方法2: 命令行参数

```bash
bash scripts/configs/autodl_config/run_autodl_experiments.sh \
    --email your_email@example.com \
    --smtp-user sender@163.com \
    --smtp-password your_authorization_code
```

### 方法3: 修改代码（不推荐）

编辑 `send_email_notification.py` 中的配置：
```python
smtp_config = {
    "server": "smtp.163.com",
    "port": 465,
    "user": "your_email@163.com",
    "password": "your_authorization_code",  # 163授权码
    "use_ssl": True
}
```

---

## 📝 获取邮箱授权码

### 163邮箱（推荐）

1. **登录163邮箱**
   - 访问 https://mail.163.com/

2. **开启SMTP服务**
   - 设置 → POP3/SMTP/IMAP
   - 开启"SMTP服务"

3. **获取授权码**
   - 点击"授权密码管理"
   - 新增授权密码
   - **记住这个授权码**（不是邮箱密码！）

### QQ邮箱

1. **登录QQ邮箱**
   - 访问 https://mail.qq.com/

2. **开启SMTP**
   - 设置 → 账户
   - 开启"POP3/SMTP服务"

3. **获取授权码**
   - 发送短信验证
   - 获取授权码

配置：
```python
smtp_config = {
    "server": "smtp.qq.com",
    "port": 465,
    "user": "your_qq@qq.com",
    "password": "authorization_code",
    "use_ssl": True
}
```

### Gmail

配置：
```python
smtp_config = {
    "server": "smtp.gmail.com",
    "port": 587,
    "user": "your_email@gmail.com",
    "password": "app_password",  # 应用专用密码
    "use_ssl": False
}
```

注意：Gmail需要开启"两步验证"并创建"应用专用密码"

---

## ⚙️ 配置说明

### 支持的SMTP服务器

| 邮箱 | SMTP服务器 | 端口 | SSL |
|------|-----------|------|-----|
| 163邮箱 | smtp.163.com | 465 | 是 |
| QQ邮箱 | smtp.qq.com | 465 | 是 |
| Gmail | smtp.gmail.com | 587 | 否(TLS) |
| 126邮箱 | smtp.126.com | 465 | 是 |
| Outlook | smtp-mail.outlook.com | 587 | 否(TLS) |

### 环境变量

```bash
# 在 ~/.bashrc 中添加（永久配置）
export SMTP_PASSWORD="your_authorization_code"

# 或在运行前临时设置
export SMTP_PASSWORD="your_authorization_code"
bash run_autodl_experiments.sh --email your@email.com
```

---

## 🧪 测试邮件功能

### 创建测试结果文件

```bash
cat > test_result.json << 'EOF'
{
    "total": 2,
    "completed": 1,
    "failed": 1,
    "start_time": "2024-10-22 10:00:00",
    "end_time": "2024-10-22 12:30:00",
    "duration_seconds": 9000,
    "successful_configs": [
        {"name": "autodl_twitter2015_deqa_seq1.json", "duration": 3600}
    ],
    "failed_configs": [
        {"name": "autodl_twitter2017_deqa_seq1.json", "error": "CUDA out of memory"}
    ]
}
EOF
```

### 测试发送

```bash
# 设置密码
export SMTP_PASSWORD="your_authorization_code"

# 发送测试邮件
python scripts/configs/autodl_config/send_email_notification.py \
    --email your_email@example.com \
    --result test_result.json
```

如果成功，你将收到一封格式化的HTML邮件。

---

## ❌ 常见问题

### Q1: 提示"未配置SMTP密码"

**原因**: 没有设置邮箱授权码

**解决**:
```bash
export SMTP_PASSWORD="your_authorization_code"
```

### Q2: "Authentication failed"

**原因**: 
- 使用了邮箱登录密码（错误）
- 应该使用授权码/应用专用密码

**解决**: 
- 重新获取授权码
- 确认使用授权码而非登录密码

### Q3: "Connection timed out"

**原因**: 
- 网络问题
- SMTP服务器或端口错误

**解决**:
- 检查网络连接
- 确认SMTP配置正确
- 尝试其他邮箱服务

### Q4: 邮件发送失败但实验继续

**说明**: 
- 这是正常行为
- 邮件发送失败不会影响实验
- 可以后续手动查看结果

**解决**:
- 检查邮件配置
- 查看结果文件: `/root/autodl-tmp/checkpoints/YYMMDD/log/autodl_result.json`

---

## 📧 邮件内容示例

### 全部成功

```
主题: ✅ AutoDL实验全部完成 (54/54)

内容:
- 总实验数: 54
- 成功: 54
- 失败: 0
- 总耗时: 27h 30m

✅ 成功完成的实验 (54个)
- autodl_twitter2015_deqa_seq1.json
- autodl_twitter2015_deqa_seq2.json
- ...
```

### 部分失败

```
主题: ⚠️ AutoDL实验完成 (50成功, 4失败)

内容:
- 总实验数: 54
- 成功: 50
- 失败: 4
- 总耗时: 26h 15m

✅ 成功完成的实验 (50个)
- ...

❌ 失败的实验 (4个)
- autodl_twitter2017_moe_seq1.json
  错误: CUDA out of memory
- ...
```

---

## 🔒 安全建议

1. **不要将授权码写入代码提交到Git**
2. **使用环境变量存储敏感信息**
3. **定期更换授权码**
4. **使用独立的邮箱账户发送通知**

---

## 🚫 禁用邮件通知

如果不需要邮件通知，运行时不指定 `--email` 参数：

```bash
bash scripts/configs/autodl_config/run_autodl_experiments.sh
# 不会发送邮件，其他功能正常
```

---

**创建日期**: 2024-10-22  
**最后更新**: 2024-10-22

