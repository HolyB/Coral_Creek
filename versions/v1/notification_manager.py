import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import json
import os
import datetime

class NotificationManager:
    def __init__(self, config_file="config.json"):
        self.config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_file)
        self.config = self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
                return {}
        return {}

    def send_email(self, subject, body, is_html=True):
        """发送邮件核心逻辑"""
        if not self.config:
            print("Notification: Config not loaded.")
            return False
            
        # 兼容旧配置字段名，优先使用新字段名
        enabled = self.config.get('email_enabled', True)
        if not enabled:
            print("Notification: Email is disabled in config.")
            return False

        sender = self.config.get('smtp_sender') or self.config.get('email_user')
        password = self.config.get('smtp_password') or self.config.get('email_pass')
        host = self.config.get('smtp_host') or self.config.get('email_host')
        port = self.config.get('smtp_port') or self.config.get('email_port')
        
        # 支持单个接收者或列表
        receivers = self.config.get('email_receivers') or self.config.get('email_to')
        if isinstance(receivers, str):
            receivers = [receivers]
            
        if not all([sender, password, host, receivers]):
            print("Notification: Missing email configuration fields.")
            return False

        message = MIMEMultipart()
        message['From'] = Header(f"StockScanner <{sender}>", 'utf-8')
        message['To'] =  Header(",".join(receivers), 'utf-8')
        message['Subject'] = Header(subject, 'utf-8')

        msg_type = 'html' if is_html else 'plain'
        message.attach(MIMEText(body, msg_type, 'utf-8'))

        try:
            port = int(port) if port else 465
            if port == 465:
                server = smtplib.SMTP_SSL(host, port)
            else:
                server = smtplib.SMTP(host, port)
                server.starttls()
            
            server.login(sender, password)
            server.sendmail(sender, receivers, message.as_string())
            server.quit()
            print(f"[OK] Notification email sent to {len(receivers)} receivers.")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to send email: {e}")
            return False

    def send_scan_report(self, market, total_scanned, blue_stocks, heima_stocks, favorites_hits=None):
        """
        发送详细的扫描报告
        blue_stocks: list of dict {'symbol':, 'name':, 'price':, 'has_day_blue': bool, ...}
        """
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        subject = f"[{market}] 股票扫描报告 - {date_str}"
        
        # HTML 样式
        style = """
        <style>
            body { font-family: Arial, sans-serif; }
            table { border-collapse: collapse; width: 100%; margin-top: 10px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .highlight { color: red; font-weight: bold; }
            .blue-tag { color: blue; font-weight: bold; }
            .heima-tag { color: purple; font-weight: bold; }
            .section { margin-top: 20px; }
        </style>
        """
        
        html = f"""
        <html>
        <head>{style}</head>
        <body>
            <h2>📊 {market} 股市扫描报告</h2>
            <p>扫描日期: <b>{date_str}</b></p>
            <p>扫描总数: {total_scanned}</p>
        """

        # 1. 自选股提醒 (优先级最高)
        if favorites_hits and len(favorites_hits) > 0:
            html += """
            <div class="section">
                <h3>⭐ 自选股信号提醒</h3>
                <table>
                    <tr><th>代码</th><th>名称</th><th>信号</th></tr>
            """
            for stock in favorites_hits:
                signals = []
                if stock.get('has_day_blue'): signals.append("<span class='blue-tag'>日线BLUE</span>")
                if stock.get('has_week_blue'): signals.append("<span class='blue-tag'>周线BLUE</span>")
                if stock.get('has_heima'): signals.append("<span class='heima-tag'>黑马</span>")
                
                html += f"<tr><td>{stock['symbol']}</td><td>{stock.get('name', '')}</td><td>{' + '.join(signals)}</td></tr>"
            html += "</table></div>"

        # 2. BLUE 信号列表
        if blue_stocks:
            html += f"""
            <div class="section">
                <h3>🔵 发现 BLUE 信号 ({len(blue_stocks)}只)</h3>
                <p>以下股票出现了日线或周线 BLUE 信号：</p>
                <table>
                    <tr><th>代码</th><th>名称</th><th>价格</th><th>信号详情</th></tr>
            """
            # 限制列表长度，防止邮件过大
            display_limit = 50
            for stock in blue_stocks[:display_limit]:
                signals = []
                if stock.get('has_day_blue'): signals.append("日线")
                if stock.get('has_week_blue'): signals.append("周线")
                
                html += f"<tr><td>{stock['symbol']}</td><td>{stock.get('name', '')}</td><td>{stock.get('price', 0)}</td><td>{'+'.join(signals)}</td></tr>"
            
            html += "</table>"
            if len(blue_stocks) > display_limit:
                html += f"<p><i>... 还有 {len(blue_stocks) - display_limit} 只未显示，请登录网页查看完整列表。</i></p>"
            html += "</div>"
        else:
            html += "<div class='section'><p>本次扫描未发现 BLUE 信号。</p></div>"

        # 3. 黑马信号列表 (简略显示)
        if heima_stocks:
            html += f"""
            <div class="section">
                <h3>🐴 发现 黑马 信号 ({len(heima_stocks)}只)</h3>
                <p>关注列表: {', '.join([s['symbol'] for s in heima_stocks[:30]])} ...</p>
            </div>
            """
        
        html += """
            <hr>
            <p>请运行 <code>streamlit run app.py</code> 查看详细图表分析。</p>
        </body>
        </html>
        """

        return self.send_email(subject, html)

if __name__ == "__main__":
    # 测试发信
    nm = NotificationManager()
    if nm.config.get('email_enabled'):
        print("Sending test email...")
        nm.send_email("StockScanner Test", "<h1>Test Success</h1><p>Email configuration is working.</p>")
    else:
        print("Email disabled in config.json")
