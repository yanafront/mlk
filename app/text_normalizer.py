from bs4 import BeautifulSoup


def clean_html(html: str) -> str:
    """
    Безопасно удаляет HTML, сохраняя текст.
    """
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator=" ", strip=True)
    return text


def normalize_vacancy(content: str) -> str:
    """
    Готовит текст вакансии для embeddings / reranking.
    """
    clean_text = clean_html(content)

    return (
        f"{clean_text}"
    )

   