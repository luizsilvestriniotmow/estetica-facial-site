import argparse
import datetime as dt
import json
import os
import re
import ssl
import sys
import urllib.parse
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
try:
    import holidays
except ImportError:
    holidays = None

def ssl_context(insecure: bool) -> ssl.SSLContext:
    if insecure:
        return ssl._create_unverified_context()
    return ssl.create_default_context()

def fetch_bytes(url: str, timeout: int = 20, *, context: ssl.SSLContext) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "daily-post-generator/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout, context=context) as res:
        return res.read()


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_rss_items(xml_bytes: bytes) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    channel = root.find("channel")
    if channel is None:
        return []

    items: list[dict] = []
    for item in channel.findall("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        description = (item.findtext("description") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()

        if not title or not link:
            continue

        items.append(
            {
                "title": title,
                "link": link,
                "description": strip_html(description),
                "pub_date": pub_date,
            }
        )
    return items

def rss_url_for_query(query: str, source: str) -> str:
    return (
        "https://news.google.com/rss/search?q="
        + urllib.parse.quote(query)
        + "&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    )

def fetch_json(url: str, *, context: ssl.SSLContext, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "daily-post-generator/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout, context=context) as res:
        return json.loads(res.read().decode("utf-8"))

def fetch_pubmed_latest(term: str, *, context: ssl.SSLContext) -> dict:
    esearch_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmax=5&sort=date&term="
        + urllib.parse.quote(term)
    )
    xml_bytes = fetch_bytes(esearch_url, timeout=30, context=context)
    root = ET.fromstring(xml_bytes)
    ids = [el.text.strip() for el in root.findall(".//IdList/Id") if el.text and el.text.strip()]
    if not ids:
        raise ValueError("PubMed: nenhum ID encontrado")

    pmid = ids[0]
    esummary_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&id="
        + urllib.parse.quote(pmid)
    )
    data = fetch_json(esummary_url, context=context)
    result = data.get("result", {}).get(pmid, {})
    title = (result.get("title") or "").strip().rstrip(".")
    pubdate = (result.get("pubdate") or "").strip()

    return {
        "title": title or f"PubMed PMID {pmid}",
        "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "description": "",
        "pub_date": pubdate,
    }

def fetch_google_news_first(query: str, *, context: ssl.SSLContext) -> dict:
    rss_url = rss_url_for_query(query, "google-news")
    rss_bytes = fetch_bytes(rss_url, context=context)
    items = parse_rss_items(rss_bytes)
    if not items:
        raise ValueError("Google News RSS: nenhum item encontrado")
    return items[0]


def build_system_message(weekday_theme: dict) -> str:
    base = (
        "Você é um redator de conteúdo para clínica premium de harmonização facial (Dra. Bruna Silvestrini). "
        "Responda SOMENTE em JSON válido (sem markdown, sem triple backticks). Use português (pt-BR). "
        "OBRIGATÓRIO: a legenda (caption) DEVE terminar com uma chamada para ação direcionando ao WhatsApp, "
        "usando o formato: '\n\n📲 Agende sua avaliação pelo WhatsApp: (11) 99550-5765\nOu clique no link da bio!'. "
        "Campos obrigatórios: source_title, source_url, caption, hashtags, image_prompt, alt_text, posting_suggestion, story_idea, disclaimer, is_video, video_script."
    )
    theme_instructions = f"\n\nINSTRUÇÕES DO TEMA DE HOJE:\n{weekday_theme['instructions']}"
    return base + theme_instructions


def call_openai(api_key: str, model: str, system_prompt: str, seed: dict, *, context: ssl.SSLContext) -> dict:
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(seed, ensure_ascii=False)},
        ],
        "temperature": 0.7,
        "response_format": { "type": "json_object" }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60, context=context) as res:
            body = res.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"OpenAI falhou: {e.read().decode('utf-8')[:200]}")
    parsed = json.loads(body)
    content = parsed.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    return json.loads(content)

def call_anthropic(api_key: str, model: str, system_prompt: str, seed: dict, *, context: ssl.SSLContext) -> dict:
    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": model, "max_tokens": 1500, "temperature": 0.7,
        "system": system_prompt,
        "messages": [{"role": "user", "content": json.dumps(seed, ensure_ascii=False)}],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60, context=context) as res:
            body = res.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Anthropic falhou: {e.read().decode('utf-8')[:200]}")
    parsed = json.loads(body)
    parts = parsed.get("content") or []
    text = "".join([p.get("text", "") for p in parts]).strip()
    # clean markdown json
    if "{" in text and "}" in text:
        text = text[text.find("{"):text.rfind("}")+1]
    return json.loads(text)

def get_available_gemini_models(api_key: str, context: ssl.SSLContext) -> list[str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={urllib.parse.quote(api_key)}"
    req = urllib.request.Request(url, headers={"content-type": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30, context=context) as res:
            data = json.loads(res.read().decode("utf-8"))
            models = data.get("models", [])
            valid_models = []
            for m in models:
                name = m.get("name", "").replace("models/", "")
                methods = m.get("supportedGenerationMethods", [])
                if "gemini" in name.lower() and "generateContent" in methods:
                    valid_models.append(name)
            return valid_models
    except Exception as e:
        print(f"Aviso interno: Falha ao listar modelos do Gemini: {e}")
        return []

def call_gemini(api_key: str, default_model: str, system_prompt: str, seed: dict, *, context: ssl.SSLContext) -> dict:
    models_to_try = [
        default_model, 
        "gemini-1.5-flash", 
        "gemini-1.5-flash-002",
        "gemini-1.5-flash-001",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro", 
        "gemini-1.5-pro-002",
        "gemini-1.5-pro-001",
        "gemini-1.0-pro",
        "gemini-pro"
    ]
    
    available = get_available_gemini_models(api_key, context)
    if available:
        print(f"🔍 Discovered Gemini models: {', '.join(available[:5])}...")
        models_to_try = available + models_to_try
    else:
        print("⚠️ Failed to dynamic list models. Relying on extensive defaults.")

    last_error = ""

    seen = set()
    unique_models = []
    for m in models_to_try:
        if m and m not in seen:
            seen.add(m)
            unique_models.append(m)

    api_versions = ["v1beta", "v1"]

    for version in api_versions:
        for m in unique_models:
            url = f"https://generativelanguage.googleapis.com/{version}/models/{urllib.parse.quote(m)}:generateContent?key={urllib.parse.quote(api_key)}"
            payload = {
                "contents": [{"role": "user", "parts": [{"text": system_prompt + "\n\n" + json.dumps(seed, ensure_ascii=False)}]}],
                "generationConfig": {"temperature": 0.7},
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=60, context=context) as res:
                    body = res.read().decode("utf-8")
                    parsed = json.loads(body)
                    candidates = parsed.get("candidates") or []
                    content = ((candidates[0] or {}).get("content") or {}) if candidates else {}
                    parts = content.get("parts") or []
                    text = "".join([p.get("text", "") for p in parts]).strip()
                    if text.startswith("```"):
                       text = re.sub(r"^```[a-zA-Z]*\n", "", text)
                       text = re.sub(r"\n```$", "", text)
                       text = text.strip()
                    
                    try:
                        return json.loads(text)
                    except Exception as json_err:
                        raise RuntimeError(f"O modelo {m} na API {version} não retornou JSON válido.")
            except urllib.error.HTTPError as e:
                err_msg = e.read().decode("utf-8")[:200]
                last_error = err_msg
                print(f"Aviso interno: Falha ao usar {m} ({version}): {err_msg}")
                if "not found" in err_msg.lower() or "not supported" in err_msg.lower() or "Method not found" in err_msg:
                    continue # Tenta o proximo modelo/versao
                raise RuntimeError(f"Gemini falhou inesperadamente em {m} ({version}): {err_msg}")
            except Exception as ex:
                 print(f"Falha ao conectar usando {m} ({version}): {ex}")
                 continue
             
    raise RuntimeError(f"Todos os modelos do Gemini falharam em v1 e v1beta! Último erro testado: {last_error}")

def ensure_fields(obj: dict) -> dict:
    required = [
        "source_title", "source_url", "caption", "hashtags", "image_prompt",
        "alt_text", "posting_suggestion", "story_idea", "disclaimer"
    ]
    for k in required:
        if k not in obj:
            obj[k] = ""
    if "is_video" not in obj:
        obj["is_video"] = False
    if "video_script" not in obj:
        obj["video_script"] = ""
        
    if not isinstance(obj.get("hashtags"), list):
        obj["hashtags"] = []
    # Force video flag true if script was generated
    if obj["video_script"].strip() != "":
        obj["is_video"] = True
    return obj


def get_theme_for_today() -> dict:
    today = dt.date.today()
    weekday = today.weekday()
    
    # 0=Monday, 6=Sunday
    theme = {}
    
    if weekday == 0:  # Segunda
        theme["name"] = "Antes e Depois - Procedimentos de Segunda"
        theme["instructions"] = (
            "Faça um post narrando um caso de 'Antes e Depois' (uma transformação) de botox ou preenchimento facial. "
            "Para o 'image_prompt', crie um cenário mostrando uma mulher sorrindo logo após o procedimento na clínica (foco no resultado lindo). "
            "A imagem não será dividida, mas sim um resultado de excelência."
        )
    elif weekday == 1: # Terça
        theme["name"] = "Notícias e Estudos (Terça)"
        theme["instructions"] = (
            "Este é o post de ciência e novidades. Baseado na notícia fornecida na seed, extraia um insight útil e simplificado para leigos sobre estética e saúde e faça um post atrativo."
        )
    elif weekday == 2: # Quarta
        theme["name"] = "Curiosidades sobre a Pele e Rosto (Quarta)"
        theme["instructions"] = (
            "Elabore uma curiosidade super interessante sobre colágeno, envelhecimento, lábios ou saúde facial."
        )
    elif weekday == 3: # Quinta
        holiday_today = False
        holiday_name = ""
        if holidays:
            br_holidays = holidays.BR()
            if today in br_holidays:
                holiday_today = True
                holiday_name = br_holidays.get(today)
                
        if holiday_today:
            theme["name"] = f"Promocional de Feriado: {holiday_name} (Quinta)"
            theme["instructions"] = (
                f"Hoje é feriado de {holiday_name}. Você deve criar uma promoção temática imperdível! "
                "OBRIGATÓRIO: A legenda deve conter EXATAMENTE a frase: 'Consulte já nossos preços promocionais! 25% em todos os procedimentos para os 10 primeiros que fecharem'. "
                "Crie uma imagem temática do feriado focada em estética com letreiros chamativos."
            )
        else:
            theme["name"] = "Ofertas Inteligentes e Site (Quinta)"
            theme["instructions"] = (
                "Post com foco comercial. Mencione e convide o usuário a conhecer mais sobre nossos procedimentos principais visitando o site https://drabrunasilvestrini.com.br. "
                "Fale de um procedimento como Fios ou Ácido Hialurônico e crie urgência de agenda."
            )
    elif weekday == 4: # Sexta
        theme["name"] = "Cuidados e Skincare (Sexta)"
        theme["instructions"] = (
            "Gere conteúdo sobre Home Care (cuidados em casa): o que usar antes ou depois dos procedimentos. Dicas de produtos (ex: protetor solar, vitamina c, hyaluronidense)."
            "Para a imagem, sugira uma cena de 'skincare routine' luxuosa com a modelo aplicando creme no rosto."
        )
    elif weekday == 5: # Sábado
        theme["name"] = "Antes e Depois - Procedimentos Avançados (Sábado)"
        theme["instructions"] = (
            "Outro caso de transformação de autoestima ('Antes e Depois' narrado). Desta vez, foque em Fios de Sustentação ou bioestimuladores e como eles promovem lifting sem cirurgia."
        )
    elif weekday == 6: # Domingo
        theme["name"] = "Mitos e Verdades - VÍDEO (Domingo)"
        theme["instructions"] = (
            "Gere o script para um VÍDEO de 'Mitos e Verdades' sobre harmonização. OBRIGATÓRIO RETORNAR json com is_video = true e video_script preenchido. "
            "No 'video_script' coloque SOMENTE o texto que será lido em voz alta pela IA (seja amigável e educativa). "
            "A 'caption' deve ser a legenda que vai no feed do Instagram instigando assistir o vídeo. "
            "O 'image_prompt' deve gerar a imagem estática de fundo para o vídeo com uma vibe de estética."
        )
    
    return theme


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="harmonização facial benefícios")
    parser.add_argument("--out-dir", default="content/daily-posts")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--anthropic-model", default=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"))
    parser.add_argument("--gemini-model", default=os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"))
    parser.add_argument("--insecure-ssl", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-any-ai", action="store_true")
    parser.add_argument("--fallback-to-draft-on-all-fail", action="store_true")
    parser.add_argument("--fallback-on-openai-error", action="store_true")
    parser.add_argument("--ai-provider-order", default="openai,anthropic,gemini")
    args = parser.parse_args()

    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    gemini_key = os.environ.get("GOOGLE_API_KEY", "").strip()

    context = ssl_context(args.insecure_ssl)

    weekday_theme = get_theme_for_today()
    print(f"🗓️ Tema do dia: {weekday_theme['name']}")

    seed = {
        "topic": "estética e harmonização facial",
        "brand": {
            "name": "Dra. Bruna Silvestrini",
            "tone": "premium e empático",
            "cta": "Agende pelo WhatsApp (11) 99550-5765",
        },
        "theme_of_today": weekday_theme["name"]
    }

    # Só buscar notícias/XML se o tema de hoje pedir (ex: Terça, onde o tema é notícias e precisamos do seed do pubmed)
    if dt.date.today().weekday() == 1:
        try:
            item = fetch_pubmed_latest(args.query, context=context)
        except Exception:
            try:
                item = fetch_google_news_first(args.query, context=context)
            except Exception:
                item = {"title": "Avanços na harmonização", "link": "https://drabrunasilvestrini.com.br", "description": ""}
        seed["article"] = item

    if args.dry_run and not openai_key:
        print("Dry-run sem execução de APIs por falta de key. Fechando.")
        return 0

    system_prompt = build_system_message(weekday_theme)

    provider_order = [p.strip().lower() for p in args.ai_provider_order.split(",") if p.strip()]
    result = None
    last_error = None

    for provider in provider_order:
        if provider == "openai" and openai_key:
            try:
                result = call_openai(api_key=openai_key, model=args.model, system_prompt=system_prompt, seed=seed, context=context)
                break
            except Exception as e:
                last_error = str(e)
        if provider == "anthropic" and anthropic_key:
            try:
                result = call_anthropic(api_key=anthropic_key, model=args.anthropic_model, system_prompt=system_prompt, seed=seed, context=context)
                break
            except Exception as e:
                last_error = str(e)
        if provider == "gemini" and gemini_key:
            try:
                result = call_gemini(api_key=gemini_key, model=args.gemini_model, system_prompt=system_prompt, seed=seed, context=context)
                break
            except Exception as e:
                last_error = str(e)

    if result is None:
        raise RuntimeError(f"Nenhum provedor de IA respondeu com sucesso. Falha: {last_error}")

    result = ensure_fields(result)

    today = dt.date.today().isoformat()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Gerar markdown bonitinho para o PR
    md_content = f"# Post Diário — {today}\n\n"
    md_content += f"**Tema do dia:** {weekday_theme['name']}\n\n"
    if result["is_video"]:
        md_content += "🎬 **VÍDEO DETECTADO** (Este post será publicado como Reels)\n\n"
        md_content += "### Roteiro de Áudio (Locução)\n"
        md_content += f"> {result['video_script']}\n\n"
    md_content += f"## Legenda\n{result['caption']}\n\n"
    md_content += f"## Hashtags\n{' '.join(['#'+h if not h.startswith('#') else h for h in result['hashtags']])}\n\n"
    md_content += f"## Imagem Prompt\n{result['image_prompt']}\n\n"
    
    out_file_md = out_dir / f"{today}.md"
    out_file_md.write_text(md_content, encoding="utf-8")

    out_file_json = out_dir / f"{today}.json"
    out_file_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Rascunho gerado em {out_file_json}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
