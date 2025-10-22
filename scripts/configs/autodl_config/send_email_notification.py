#!/usr/bin/env python3
"""
AutoDL实验完成邮件通知脚本

使用方法:
    python send_email_notification.py --email your@email.com --result result.json
"""

import json
import argparse
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime


def send_email(to_email: str, subject: str, html_content: str, smtp_config: dict = None):
    """
    发送邮件
    
    Args:
        to_email: 收件人邮箱
        subject: 邮件主题
        html_content: HTML内容
        smtp_config: SMTP配置（如果不提供则使用默认配置）
    """
    # 默认SMTP配置（使用163邮箱）
    if smtp_config is None:
        smtp_config = {
            "server": "smtp.163.com",
            "port": 465,
            "user": "15932448905@163.com",  # 需要替换为实际邮箱
            "password": "FC4Qx36H9L8hd2SS",  # 需要设置授权码
            "use_ssl": True
        }
    
    # 如果没有配置密码，尝试从环境变量读取
    if not smtp_config.get("password"):
        import os
        smtp_config["password"] = os.environ.get("SMTP_PASSWORD", "")
    
    if not smtp_config["password"]:
        print("警告: 未配置SMTP密码，邮件发送功能将被禁用")
        print("解决方法:")
        print("  1. 设置环境变量: export SMTP_PASSWORD=your_password")
        print("  2. 或修改 send_email_notification.py 中的密码")
        return False
    
    try:
        # 创建邮件
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = smtp_config['user']
        msg['To'] = to_email
        
        # 添加HTML内容
        html_part = MIMEText(html_content, 'html', 'utf-8')
        msg.attach(html_part)
        
        # 发送邮件
        if smtp_config.get('use_ssl'):
            server = smtplib.SMTP_SSL(smtp_config['server'], smtp_config['port'])
        else:
            server = smtplib.SMTP(smtp_config['server'], smtp_config['port'])
            server.starttls()
        
        server.login(smtp_config['user'], smtp_config['password'])
        server.send_message(msg)
        server.quit()
        
        print(f"✓ 邮件已发送到: {to_email}")
        return True
        
    except Exception as e:
        print(f"✗ 邮件发送失败: {e}")
        return False


def generate_email_content(result_data: dict) -> tuple:
    """
    生成邮件内容
    
    Returns:
        (subject, html_content)
    """
    total = result_data.get('total', 0)
    completed = result_data.get('completed', 0)
    failed = result_data.get('failed', 0)
    successful_configs = result_data.get('successful_configs', [])
    failed_configs = result_data.get('failed_configs', [])
    start_time = result_data.get('start_time', 'N/A')
    end_time = result_data.get('end_time', 'N/A')
    duration = result_data.get('duration_seconds', 0)
    
    # 计算时间
    hours = duration // 3600
    minutes = (duration % 3600) // 60
    
    # 邮件主题
    if failed == 0:
        subject = f"✅ AutoDL实验全部完成 ({completed}/{total})"
    else:
        subject = f"⚠️ AutoDL实验完成 ({completed}成功, {failed}失败)"
    
    # HTML内容
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; padding: 30px; border-radius: 10px; text-align: center; }}
        .summary {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .summary-item {{ display: inline-block; margin: 10px 20px; }}
        .summary-label {{ font-weight: bold; color: #666; }}
        .summary-value {{ font-size: 24px; font-weight: bold; }}
        .success {{ color: #28a745; }}
        .failed {{ color: #dc3545; }}
        .section {{ margin: 30px 0; }}
        .section-title {{ font-size: 20px; font-weight: bold; margin-bottom: 15px; 
                         border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
        .config-list {{ background: white; border: 1px solid #dee2e6; border-radius: 5px; }}
        .config-item {{ padding: 10px 15px; border-bottom: 1px solid #dee2e6; }}
        .config-item:last-child {{ border-bottom: none; }}
        .config-name {{ font-weight: bold; }}
        .error-msg {{ color: #dc3545; font-size: 12px; margin-top: 5px; 
                     background: #fff3cd; padding: 8px; border-radius: 4px; }}
        .footer {{ text-align: center; color: #666; margin-top: 40px; padding-top: 20px; 
                  border-top: 1px solid #dee2e6; }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; 
                 font-size: 12px; font-weight: bold; }}
        .badge-success {{ background: #28a745; color: white; }}
        .badge-danger {{ background: #dc3545; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎉 AutoDL实验完成通知</h1>
            <p>持续学习实验已全部完成</p>
        </div>
        
        <div class="summary">
            <div class="summary-item">
                <div class="summary-label">总实验数</div>
                <div class="summary-value">{total}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">成功</div>
                <div class="summary-value success">{completed}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">失败</div>
                <div class="summary-value failed">{failed}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">总耗时</div>
                <div class="summary-value">{hours}h {minutes}m</div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">⏰ 时间统计</div>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background: #f8f9fa;">
                    <td style="padding: 10px; border: 1px solid #dee2e6;"><strong>开始时间</strong></td>
                    <td style="padding: 10px; border: 1px solid #dee2e6;">{start_time}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #dee2e6;"><strong>结束时间</strong></td>
                    <td style="padding: 10px; border: 1px solid #dee2e6;">{end_time}</td>
                </tr>
                <tr style="background: #f8f9fa;">
                    <td style="padding: 10px; border: 1px solid #dee2e6;"><strong>总时长</strong></td>
                    <td style="padding: 10px; border: 1px solid #dee2e6;">{hours}小时 {minutes}分钟</td>
                </tr>
            </table>
        </div>
"""
    
    # 成功的配置
    if successful_configs:
        html += f"""
        <div class="section">
            <div class="section-title">✅ 成功完成的实验 ({len(successful_configs)}个)</div>
            <div class="config-list">
"""
        for cfg in successful_configs:
            html += f"""
                <div class="config-item">
                    <span class="badge badge-success">SUCCESS</span>
                    <span class="config-name">{cfg['name']}</span>
                    <span style="color: #666; font-size: 12px;"> - 耗时: {cfg.get('duration', 0)}秒</span>
                </div>
"""
        html += """
            </div>
        </div>
"""
    
    # 失败的配置
    if failed_configs:
        html += f"""
        <div class="section">
            <div class="section-title">❌ 失败的实验 ({len(failed_configs)}个)</div>
            <div class="config-list">
"""
        for cfg in failed_configs:
            error_msg = cfg.get('error', 'Unknown error')
            # 截断过长的错误消息
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            
            html += f"""
                <div class="config-item">
                    <span class="badge badge-danger">FAILED</span>
                    <span class="config-name">{cfg['name']}</span>
                    <div class="error-msg">
                        <strong>错误:</strong> {error_msg}
                    </div>
                </div>
"""
        html += """
            </div>
        </div>
"""
    
    # 页脚
    html += """
        <div class="footer">
            <p>本邮件由AutoDL实验系统自动发送</p>
            <p style="font-size: 12px; color: #999;">
                如有问题，请检查服务器日志: /root/autodl-tmp/checkpoints/*/log/
            </p>
        </div>
    </div>
</body>
</html>
"""
    
    return subject, html


def main():
    parser = argparse.ArgumentParser(description='发送AutoDL实验完成通知邮件')
    parser.add_argument('--email', type=str, required=True, help='收件人邮箱')
    parser.add_argument('--result', type=str, required=True, help='结果JSON文件路径')
    parser.add_argument('--smtp-server', type=str, default='smtp.163.com', help='SMTP服务器')
    parser.add_argument('--smtp-port', type=int, default=465, help='SMTP端口')
    parser.add_argument('--smtp-user', type=str, help='SMTP用户名')
    parser.add_argument('--smtp-password', type=str, help='SMTP密码')
    
    args = parser.parse_args()
    
    # 读取结果数据
    result_file = Path(args.result)
    if not result_file.exists():
        print(f"错误: 结果文件不存在: {args.result}")
        return 1
    
    with open(result_file, 'r', encoding='utf-8') as f:
        result_data = json.load(f)
    
    # 生成邮件内容
    subject, html_content = generate_email_content(result_data)
    
    # SMTP配置
    smtp_config = None
    if args.smtp_user:
        smtp_config = {
            "server": args.smtp_server,
            "port": args.smtp_port,
            "user": args.smtp_user,
            "password": args.smtp_password or "",
            "use_ssl": True
        }
    
    # 发送邮件
    if send_email(args.email, subject, html_content, smtp_config):
        print("邮件发送成功！")
        return 0
    else:
        print("邮件发送失败")
        return 1


if __name__ == "__main__":
    exit(main())

