<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=00bfbf&height=120&section=header" alt="Cabeçalho decorativo"/>

<h1 align="center">
  <img src="https://fit-tecnologia.org.br/ava/pluginfile.php/1/theme_moove/logo/1784901257/pnaat-positivo.png" width="300px" alt="PNAAT"/>
  <br/>
  Desafio Técnico – Edge AI com Visão Computacional
</h1>

<p align="center">
  Classificação de dígitos manuscritos do MNIST com uma rede neural
  convolucional, conversão para TensorFlow Lite e inferência otimizada para
  dispositivos Edge.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white"/>
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.16.1-FF6F00?logo=tensorflow&logoColor=white"/>
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-1.26.4-013243?logo=numpy&logoColor=white"/>
  <img alt="Status" src="https://img.shields.io/badge/status-em%20validação-yellow"/>
</p>

---

## 👤 Identificação

<table>
  <tr>
    <td align="center">
      <img src="https://avatars.githubusercontent.com/u/86336670?v=4" width="100px" alt="Mateus Alencar Ferreira"/>
      <br/>
      <strong>Mateus Alencar Ferreira</strong>
      <br/>
      <a href="https://github.com/ferreiramateusalencar">@ferreiramateusalencar</a>
      <br/><br/>
      <a href="https://www.linkedin.com/in/mateus-alencar-ferreira/" title="LinkedIn">🌐 LinkedIn</a>
    </td>
  </tr>
</table>

| Campo | Informação |
| --- | --- |
| Projeto escolhido | Projeto 1 — Classificação MNIST |
| Tarefa | Classificação de dígitos manuscritos de 0 a 9 |
| Artefato treinado | `model.h5` |
| Artefato Edge | `model.tflite` |

---

## 1️⃣ Resumo da Arquitetura do Modelo

A CNN implementada em `train_model.py` recebe imagens em escala de cinza com
formato **`(28, 28, 1)`**. Antes do treinamento, os pixels são convertidos para
`float32` e normalizados do intervalo original `[0, 255]` para **`[0, 1]`**.

O conjunto original de treinamento do MNIST é dividido explicitamente em:

- **54.000 imagens para treinamento**;
- **6.000 imagens para validação**.

### Blocos convolucionais

O modelo possui três blocos de extração de características:

| Bloco | Filtros | Estrutura |
| ---: | ---: | --- |
| 1 | 16 | `Conv2D(3×3)` → `BatchNormalization` → ReLU → `MaxPooling2D` |
| 2 | 32 | `Conv2D(3×3)` → `BatchNormalization` → ReLU → `MaxPooling2D` |
| 3 | 64 | `Conv2D(3×3)` → `BatchNormalization` → ReLU → `MaxPooling2D` |

As camadas convolucionais aprendem características progressivamente mais
complexas. Os primeiros filtros identificam bordas e traços simples; os blocos
seguintes combinam essas informações para reconhecer curvas e formas
características dos dígitos.

`BatchNormalization` estabiliza as ativações e contribui para um treinamento
mais consistente. `MaxPooling2D` reduz as dimensões espaciais dos mapas de
características, diminuindo o custo computacional e preservando as informações
mais relevantes.

### Camadas de classificação

Depois dos blocos convolucionais, o modelo utiliza:

- **`Flatten`** para transformar os mapas de características em um vetor;
- **`Dense(64, ReLU)`** para combinar as características aprendidas;
- **`Dropout(0.30)`** para reduzir o risco de overfitting;
- **`Dense(10, Softmax)`** para gerar a probabilidade de cada dígito de 0 a 9.

```text
Entrada (28×28×1)
    ↓
Conv2D(16) → BatchNorm → ReLU → MaxPooling
    ↓
Conv2D(32) → BatchNorm → ReLU → MaxPooling
    ↓
Conv2D(64) → BatchNorm → ReLU → MaxPooling
    ↓
Flatten → Dense(64, ReLU) → Dropout(30%)
    ↓
Dense(10, Softmax)
```

### Estratégia de treinamento

| Hiperparâmetro | Configuração |
| --- | --- |
| Otimizador | Adam |
| Learning rate | `0.001` |
| Máximo de épocas | 12 |
| Batch size | 128 |
| Early stopping | `val_loss`, paciência de 2 épocas |
| Melhor peso | Restaurado automaticamente |
| Dispositivo | CPU |
| Semente aleatória | 42 |

O **Dropout de 30%** foi escolhido para regularizar a rede sem eliminar uma
quantidade excessiva de informações. O **batch size de 128** reduz o tempo de
treinamento em CPU e ainda mantém atualizações frequentes dos gradientes. O
limite de 12 épocas, combinado ao `EarlyStopping`, evita processamento
desnecessário quando a perda de validação deixa de melhorar.

---

## 2️⃣ Bibliotecas Utilizadas

| Biblioteca | Versão | Utilização |
| --- | ---: | --- |
| Python | 3.10 | Execução dos scripts e ambiente da CI |
| TensorFlow | 2.16.1 | Treinamento, salvamento e conversão TFLite |
| `tf.keras` | TensorFlow 2.16.1 | Construção da CNN e early stopping |
| NumPy | 1.26.4 | Preparação das imagens e interpretação das predições |
| `os` | Biblioteca padrão | Caminhos e medição dos arquivos |

As dependências principais estão fixadas em `requirements.txt`. As versões do
ambiente ativo podem ser conferidas com:

```bash
python --version
pip show tensorflow keras numpy
```

---

## 3️⃣ Técnica de Otimização do Modelo

No arquivo `optimize_model.py` é aplicada a **Quantização de Faixa Dinâmica**
(*Dynamic Range Quantization*):

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

Durante a conversão, os pesos originalmente armazenados em `float32` são
geralmente representados em `int8`. As entradas e saídas permanecem em ponto
flutuante, permitindo que o script de inferência continue recebendo imagens
normalizadas em `float32`.

A técnica foi escolhida porque:

- reduz o tamanho do modelo;
- diminui a memória necessária para armazenar os pesos;
- não exige um conjunto representativo de calibração;
- normalmente preserva a acurácia em modelos compactos de classificação;
- facilita a implantação em equipamentos com recursos limitados.

Depois da conversão, o script carrega `model.tflite` com
`tf.lite.Interpreter` e aloca os tensores, confirmando que o artefato gerado
pode ser interpretado pelo runtime do TensorFlow Lite.

---

## 4️⃣ Resultados Obtidos

> [!IMPORTANT]
> Os artefatos devem ser regenerados com o `train_model.py` atual antes da
> entrega definitiva. Os valores de acurácia e inferência serão preenchidos
> exclusivamente com a saída da nova execução.

| Métrica | Valor |
| --- | ---: |
| Acurácia final de validação | **Pendente de novo treinamento** |
| Acurácia no validador oficial | **Pendente de validação** |
| Tamanho atual de `model.h5` | 1.165.592 bytes — 1.138,3 KB |
| Tamanho atual de `model.tflite` | 104.200 bytes — 101,8 KB |
| Redução atual de tamanho | 1.061.392 bytes — 91,1% |

Os tamanhos acima foram medidos diretamente nos arquivos presentes no
repositório. Eles deverão ser atualizados caso o novo treinamento e a nova
conversão produzam arquivos com dimensões diferentes.

---

## 5️⃣ Comentários Adicionais

### Dificuldade encontrada

O principal desafio é equilibrar capacidade de aprendizado, tempo de
treinamento em CPU e tamanho final do modelo. Uma arquitetura muito pequena
pode perder acurácia, enquanto uma arquitetura excessivamente profunda aumenta
o custo computacional sem benefício proporcional para o MNIST.

### Decisões técnicas

- Foram utilizados três blocos convolucionais para atender aos requisitos e
  permitir aprendizado progressivo de características.
- A quantidade de filtros aumenta de 16 para 64 conforme as dimensões espaciais
  são reduzidas.
- O bias das convoluções foi desativado porque o deslocamento já é tratado pela
  Batch Normalization.
- O treinamento é limitado a 12 épocas, mas pode terminar antes pelo
  `EarlyStopping`.
- A GPU é desabilitada explicitamente para garantir aderência ao requisito de
  treinamento em CPU.
- A semente 42 melhora a reprodutibilidade dos resultados.

### Limitações

O modelo aprende a partir de imagens centralizadas e normalizadas do MNIST. Ele
não deve ser considerado confiável para números fotografados em papéis, placas
ou ambientes reais sem novo treinamento com dados representativos dessas
condições.

### Aprendizados

O projeto demonstra o fluxo completo:

**treinamento → validação → salvamento → quantização → inferência TFLite**

Também evidencia que otimizar para Edge AI não significa apenas reduzir o
tamanho do arquivo. É necessário verificar a métrica de validação e confirmar
que o modelo convertido ainda executa inferências coerentes.

---

## 6️⃣ Exemplo de Inferência

A inferência é executada especificamente com o modelo otimizado:

```bash
python run_inference.py
```

O script analisa dez amostras do conjunto de teste e apresenta classe predita,
classe real e confiança:

```text
Inferência de 10 amostras com model.tflite
Amostra 01: predito=<resultado> | real=<rótulo> | confiança=<valor>
Amostra 02: predito=<resultado> | real=<rótulo> | confiança=<valor>
Amostra 03: predito=<resultado> | real=<rótulo> | confiança=<valor>
Amostra 04: predito=<resultado> | real=<rótulo> | confiança=<valor>
Amostra 05: predito=<resultado> | real=<rótulo> | confiança=<valor>
Amostra 06: predito=<resultado> | real=<rótulo> | confiança=<valor>
Amostra 07: predito=<resultado> | real=<rótulo> | confiança=<valor>
Amostra 08: predito=<resultado> | real=<rótulo> | confiança=<valor>
Amostra 09: predito=<resultado> | real=<rótulo> | confiança=<valor>
Amostra 10: predito=<resultado> | real=<rótulo> | confiança=<valor>
Acertos na amostra: <total>/10
```

> [!NOTE]
> Este bloco deverá ser substituído pela saída integral do terminal depois que
> `model.h5` e `model.tflite` forem regenerados. O comentário sobre os acertos
> ou erros deve refletir o resultado realmente observado.

---

## ▶️ Como Executar

Na pasta do projeto:

```bash
pip install -r requirements.txt
python train_model.py
python optimize_model.py
python run_inference.py
```

Para executar o validador oficial a partir da raiz do repositório:

```bash
python .github/scripts/validate_1_mnist.py projetos/1-classificacao-mnist
```

---

## 📄 Licença

Este projeto é disponibilizado sob a licença MIT. Consulte o arquivo
[LICENSE](../../LICENSE) para os termos completos.

<p align="center">
  <strong>Projeto desenvolvido para o Processo Seletivo Intensivo Maker — 2ª Região</strong>
</p>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=00bfbf&height=120&section=footer" alt="Rodapé decorativo"/>
