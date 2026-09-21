from urllib.parse import urlparse

SPECIAL_CHARS = {
    'dot': '.', 'hyphen': '-', 'underline': '_', 'slash': '/',
    'questionmark': '?', 'equal': '=', 'at': '@', 'and': '&',
    'exclamation': '!', 'space': ' ', 'tilde': '~', 'comma': ',',
    'plus': '+', 'asterisk': '*', 'hashtag': '#', 'dollar': '$',
    'percent': '%',
}


def split_url(url):
    """Devolve (domain, directory, file, params).
    Pedaço ausente -> string vazia."""
    # se a URL não tiver esquema (http://), adicione um,
    # senão o urlparse não reconhece o domínio.
    if '://' not in url:
        url = 'http://' + url

    # use urlparse para pegar netloc, path e query.
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    params = parsed.query


    directory, _, file = path.rpartition('/')
    directory = directory.strip('/')

    return domain, directory, file, params


def count_chars(text, suffix):
    """Conta os 17 caracteres especiais em `text`.
    Chaves no formato 'qty_{nome}_{suffix}'.
    Se text for vazio, todos os valores devem ser -1."""
    
    if not text:
        return {f'qty_{nome}_{suffix}': -1 for nome in SPECIAL_CHARS}
    return {f'qty_{nome}_{suffix}': text.count(char) for nome, char in SPECIAL_CHARS.items()}


if __name__ == '__main__':
    testes = [
        'https://www.google.com',
        'http://site.com/a/b/login.php?user=1&x=2',
        'site.com/login',
    ]
    for u in testes:
        print(u, '->', split_url(u))