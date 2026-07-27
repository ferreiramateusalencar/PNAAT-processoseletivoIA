# Processo Seletivo – Intensivo Maker | AI

Bem-vindo(a) à **etapa prática do processo seletivo para o Intensivo Maker**.

Esta atividade tem como objetivo avaliar competências técnicas relacionadas a **Machine Learning**, **Visão Computacional** e **Otimização de modelos para sistemas embarcados (Edge AI)**, a partir da aplicação prática dos conhecimentos adquiridos nos cursos EAD da etapa anterior.

> 🎯 **Importante**
> O foco deste desafio é avaliar sua capacidade de **projetar, treinar e otimizar um modelo de IA** — e de **entregar corretamente** os artefatos gerados.

---

## 📌 Navegação Rápida

- 🏁 [Passo 0 – Antes de Tudo](#-passo-0-antes-de-tudo)
- ⚙ [Passo 1 – Preparando o Ambiente](#-passo-1-preparando-o-ambiente)
- 🧭 [Passo 2 – Escolha do Projeto](#-passo-2-escolha-do-projeto)
- 📤 [Passo 3 – Instruções de Entrega](#-passo-3-instruções-de-entrega)
- ⚠️ [Restrições Gerais de Engenharia](#️-restrições-gerais-de-engenharia)
- 🆘 [Suporte](#-suporte)

---

## 🏁 Passo 0: Antes de Tudo

Caso você **nunca tenha utilizado Git ou GitHub**, não se preocupe. Siga atentamente as etapas abaixo.

### 1️⃣ Criação de Conta no GitHub

1. Acesse: https://github.com
2. Clique em **Sign up**
3. Crie sua conta gratuita seguindo as instruções da plataforma

(*O GitHub será utilizado para envio, versionamento e correção automática do seu projeto.*)

### 2️⃣ Instalação do Git

O **Git** é a ferramenta que permite versionar e enviar seu código para o GitHub.

- **Windows** — Baixe e instale o **Git Bash**: https://git-scm.com/downloads
- **Linux / macOS** — Verifique se o Git já está instalado:
  ```bash
  git --version
  ```

---

## ⚙ Passo 1: Preparando o Ambiente

Para desenvolver o desafio, você deverá criar uma cópia deste repositório.

### 1️⃣ Fork do Repositório

No canto superior direito desta página, clique em **Fork**. Uma cópia deste repositório será criada no **seu perfil do GitHub**.
(*O Fork permite que você trabalhe de forma independente sem alterar o repositório original.*)

### 2️⃣ Clone do Repositório

No repositório do **seu Fork**, clique em **<> Code**, copie a URL e execute:

```bash
git clone https://github.com/SEU_USUARIO/nome-do-repositorio.git
cd nome-do-repositorio
```

### 3️⃣ Preparação do Ambiente de Execução

Você pode executar o projeto de **três formas**. Escolha apenas uma.

#### Opção A – Ambiente Python Local
Requisitos: Python **3.10 ou 3.11** e pip.

As dependências ficam dentro da pasta do projeto escolhido (veja Passo 2), então instale-as **depois** de escolher seu projeto:

```bash
cd projetos/<pasta-do-projeto-escolhido>
pip install -r requirements.txt
```

#### Opção B – Dev Container
Este repositório inclui um **Dev Container** para facilitar a criação de um ambiente Python padronizado.

**Requisitos:** VS Code, Docker instalado, extensão **Dev Containers**.

**Passos:** abra o repositório no VS Code → **"Reopen in Container"** → aguarde a criação automática do ambiente.

#### Opção C – via browser (GitHub Codespaces)
1. Clique em **<> Code**
2. Clique em **Codespaces**
3. Clique em **Create codespace on main**

> Será aberta uma instância do VS Code no seu navegador com o container configurado.

---

## 🧭 Passo 2: Escolha do Projeto

Este desafio oferece **três opções de projeto**, todas em Visão Computacional e com **níveis de dificuldade equivalentes**. Você deve escolher **apenas uma**.

| # | Projeto | Tarefa | Dataset |
|---|---------|--------|---------|
| 1 | [Classificação MNIST](projetos/1-classificacao-mnist/README.md) | Classificação de dígitos manuscritos (0-9) | `tf.keras.datasets.mnist` |
| 2 | [Classificação CIFAR-10](projetos/2-classificacao-cifar/README.md) | Classificação de imagens coloridas (10 categorias de objetos/animais) | `tf.keras.datasets.cifar10` |
| 3 | [Detecção de Máscaras Faciais](projetos/3-deteccao-mascaras/README.md) | Detecção de objetos: localizar rostos e classificar uso de máscara (fine-tuning de YOLO) | Face Mask Detection (Kaggle, CC0) — já incluso no repositório |

Clique no link do projeto escolhido para ver as instruções técnicas completas e o template do relatório.

### ⚠️ Depois de escolher, você DEVE:

1. Trabalhar **apenas** dentro da pasta do projeto escolhido (`projetos/N-nome-do-projeto/`).
2. **Apagar as pastas dos outros dois projetos** dentro de `projetos/` antes do commit final.
3. Manter os nomes de arquivos e a estrutura interna da pasta do projeto **sem alterações**.

> 🤖 **Por quê apagar as outras pastas?**
> A correção automática (GitHub Actions) identifica qual projeto você escolheu verificando qual pasta restou dentro de `projetos/`. Se mais de uma pasta permanecer (ou nenhuma), a validação falha automaticamente com uma mensagem explicando o problema.

---

## 📤 Passo 3: Instruções de Entrega

### ✔️ Antes de enviar

Dentro da pasta do seu projeto, execute os scripts e confirme que os arquivos foram gerados:

```bash
cd projetos/<pasta-do-projeto-escolhido>
python train_model.py       # deve gerar model.h5 (Projetos 1 e 2) ou model.pt (Projeto 3)
python optimize_model.py    # deve gerar model.tflite
```

> ⚠️ **Importante:** a correção automática **não treina nada por você**. Ela valida os artefatos que **você gerou localmente e enviou (commitou) para o repositório**. Se esses arquivos não estiverem no seu commit, a validação falha.

### ⬆️ Envio do Código

```bash
git add .
git commit -m "Entrega do desafio técnico - Seu Nome"
git push origin main
```

### 🔍 Verificação Automática

1. Acesse a aba **Actions** no GitHub do seu Fork
2. Verifique se o workflow foi executado com sucesso (✅)
3. Em caso de erro (❌), consulte os logs, corrija e envie novamente

### 📎 Submissão Final

Copie o link do seu repositório e envie conforme orientações do processo seletivo no Moodle.

---

## ⚠️ Restrições Gerais de Engenharia

Válidas para os três projetos (detalhes específicos estão no README de cada um):

- Treinamento apenas em **CPU**
- Sem uso de modelos pré-treinados — **exceto no Projeto 3**, onde o fine-tuning
  de um modelo pré-treinado (YOLO11n) é intencional e faz parte do desafio
- Número de épocas limitado (compatível com execução rápida — exceto o Projeto 3,
  que naturalmente leva mais tempo por envolver fine-tuning de um detector)
- Código deve executar do início ao fim **sem intervenção manual**
- Os artefatos do modelo treinado e do modelo otimizado (`model.h5`/`model.pt` e
  `model.tflite`, dependendo do projeto) **devem ser gerados localmente e
  enviados (commitados) junto com o código** — a correção automática apenas os
  valida, não os gera

> **Importante:** o objetivo não é obter a maior acurácia possível, mas sim demonstrar **engenharia eficiente** e a capacidade de entregar um pipeline completo e reprodutível.

---

## 📚 Material de Apoio

Os cursos realizados na etapa anterior **devem ser utilizados como referência**:

- 📘 Fundamentos de Inteligência Artificial para Sistemas Embarcados
- 👁️ Sistemas de Visão Computacional Embarcada
- ⚙️ Otimização de Modelos em Sistemas Embarcados

---

## 🆘 Suporte

Em caso de dúvidas:

- Consulte o material dos cursos EAD
- Leia atentamente este README e o README do projeto escolhido
- Analise os logs das GitHub Actions
- Utilize os canais oficiais para contato com os instrutores

Boa sorte no processo seletivo.
****
