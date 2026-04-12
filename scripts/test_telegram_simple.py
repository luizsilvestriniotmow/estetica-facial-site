import os
import json
import urllib.request
import ssl

def test_telegram():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("Erro: TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID não configurados.")
        return

    ctx = ssl._create_unverified_context()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": "✅ **Teste de Conexão: Jarbas está online!**\nA comunicação entre o GitHub e o Telegram está funcionando perfeitamente.",
        "parse_mode": "Markdown"
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, context=ctx) as res:
            print("Mensagem de teste enviada com sucesso!")
    except Exception as e:
        print(f"Falha ao enviar mensagem de teste: {e}")

if __name__ == "__main__":
    test_telegram()
