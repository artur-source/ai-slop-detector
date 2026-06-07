# ai-slop-detector

Detecte texto gerado por IA usando DistilBERT fine-tunado + OpenAI API

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-black)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License MIT](https://img.shields.io/badge/License-MIT-green)
![pytest](https://img.shields.io/badge/tests-pytest-blue)

## O que eu construí

Este projeto é uma aplicação web Flask que executa dois detectores de texto gerado por IA lado a lado.
O detector local usa DistilBERT fine-tunado no dataset HC3, comparando respostas humanas com respostas do ChatGPT.
O detector via OpenAI usa `gpt-4o-mini` como comparação de referência.
O usuário pode colar texto diretamente ou enviar uma URL, que é raspada automaticamente antes da análise.

## Como funciona

```text
User input (text or URL)
       |
       v
   Scraper (BeautifulSoup)        [if URL]
       |
       v
   Text validation & cleanup
       /              \
Local DistilBERT    OpenAI gpt-4o-mini
       \              /
   Side-by-side results page
```

## Performance do Modelo

| Métrica   | Local DistilBERT | OpenAI gpt-4o-mini |
|-----------|------------------|--------------------|
| Accuracy  | TBD              | TBD                |
| Precision | TBD              | TBD                |
| Recall    | TBD              | TBD                |
| F1        | TBD              | TBD                |

Os resultados serão atualizados após o treino completo no Google Colab (GPU T4).

## Estrutura do Projeto

```text
ai-slop-detector/
+-- app/
|   +-- __init__.py              # Factory da aplicação Flask
|   +-- routes.py                # Rotas web para envio do formulário e resultados
|   +-- detector.py              # Motor de detecção com DistilBERT local e OpenAI
|   +-- scraper.py               # Utilitários de scraping de URL e limpeza de texto
|   +-- templates/
|       +-- index.html           # Formulário principal de entrada
|       +-- result.html          # Página de resultados lado a lado
+-- model/
|   +-- train_classifier.py      # Script de carregamento do HC3 e fine-tuning do DistilBERT
|   +-- evaluate.py              # Script de métricas de avaliação e matriz de confusão
+-- data/
|   +-- README.md                # Notas de download do dataset HC3
+-- assets/                      # Gráficos gerados e assets visuais
+-- tests/
|   +-- test_detector.py         # Testes unitários do comportamento de detecção
|   +-- test_scraper.py          # Testes unitários de scraping e extração de texto
+-- config.py                    # Configuração compartilhada do projeto
+-- requirements.txt             # Dependências Python
+-- .gitignore                   # Segredos, caches, checkpoints e modelos salvos ignorados
+-- .env.example                 # Template de variáveis de ambiente
+-- README.md                    # Documentação em inglês
+-- README.pt.md                 # Documentação em português
+-- run.py                       # Entrypoint Flask local
```

## Como Rodar

### Local

```bash
git clone https://github.com/SEU_USUARIO/ai-slop-detector
cd ai-slop-detector
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your OPENAI_API_KEY
python model/train_classifier.py   # train the local model first
python run.py
```

### Testes

```bash
pytest tests/ -v
```

## Decisões de Design

- Detecção dupla: comparar modelo local com API expõe trade-offs de custo e acurácia.
- Carregamento preguiçoso do modelo: o DistilBERT carrega apenas na primeira requisição.
- Scraper com headers reais: ajuda a evitar bloqueios em sites comuns.

## Referências

- HC3 Dataset: https://huggingface.co/datasets/Hello-SimpleAI/HC3
- DistilBERT: https://arxiv.org/abs/1910.01108
- Hello, GPT-4o-mini: https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/

Construído por Artur como projeto de portfólio - Análise e Desenvolvimento de Sistemas, UniPiaget
