# AutoDL快速开始（5分钟搞定）

## 🎯 一分钟理解

- **存储**: `/root/autodl-tmp/checkpoints/日期/` (数据盘)
- **代码**: `/root/MCM/` (系统盘)
- **通知**: 邮件自动发送结果
- **关机**: 完成后自动关机

---

## 🚀 三步运行

### 1️⃣ 本地生成配置

```bash
python scripts/generate_autodl_configs.py
```

### 2️⃣ AutoDL设置邮件

```bash
# 获取163邮箱授权码（见下方）
export SMTP_PASSWORD="授权码"
```

### 3️⃣ 运行实验

```bash
cd /root/MCM
nohup bash scripts/configs/autodl_config/run_autodl_experiments.sh \
    --email your@email.com > run.log 2>&1 &
```

**完成！** 退出SSH，等邮件通知即可。

---

## 📧 获取163邮箱授权码（2分钟）

1. 打开 https://mail.163.com/
2. 点击 **设置** → **POP3/SMTP/IMAP**
3. 开启 **SMTP服务**
4. 点击 **授权密码管理** → **新增授权密码**
5. 按提示发送短信
6. **复制授权码**（16位字符）

```bash
# 在AutoDL上设置
export SMTP_PASSWORD="刚才复制的授权码"
```

---

## 📊 监控进度

```bash
# 查看进度
bash scripts/configs/autodl_config/check_progress.sh

# 查看日志
tail -f /root/autodl-tmp/checkpoints/*/log/autodl_run_*.log

# GPU状态
nvidia-smi
```

---

## ⚠️ 重要提醒

### ✅ 做这些

- ✅ 使用 **授权码** 而非邮箱密码
- ✅ 用 nohup 后台运行
- ✅ 退出SSH前确认进程在running
- ✅ 记得下载结果

### ❌ 不要做

- ❌ 不要用邮箱登录密码
- ❌ 不要直接运行（会断开）
- ❌ 不要关闭自动关机
- ❌ 不要忘记下载checkpoint

---

## 🆘 出问题了？

### 邮件发送失败

```bash
# 检查授权码
echo $SMTP_PASSWORD

# 测试邮件
python scripts/configs/autodl_config/send_email_notification.py \
    --email your@email.com --result /tmp/test.json
```

### 进程没运行

```bash
# 检查进程
ps aux | grep train_with_zero_shot

# 查看最新日志
tail -100 /root/autodl-tmp/checkpoints/*/log/autodl_run_*.log
```

### 空间不足

```bash
# 检查磁盘
df -h

# 数据应该在数据盘
/root/autodl-tmp/  # 100GB+
```

---

## 📞 更多帮助

- 详细教程: `AUTODL_SETUP_GUIDE.md`
- 邮件配置: `EMAIL_SETUP.md`
- 更新说明: `AUTODL_V2_UPDATES.md`

---

**5分钟上手，27小时自动完成，邮件通知结果！** 🎉







