<img width=100% src="https://capsule-render.vercel.app/api?type=waving&color=00bfbf&height=120&section=header"/>
<h1 align="center"><img align="center" style="border-radius: 100%;" src="https://fit-tecnologia.org.br/ava/pluginfile.php/1/theme_moove/logo/1784901257/pnaat-positivo.png" width="300px;" alt=""><br>Desafio Técnico – Edge AI com Visão Computacional</h1>

👤 Identificação
<table>
  <tr>
    <td align="center"><img style="border-radius: 50%;" src="https://avatars.githubusercontent.com/u/86336670?v=4" width="100px;"/><br/><a href="https://github.com/ferreiramateusalencar">Mateus A. Ferreira<a/><br/><br/><a href="https://www.linkedin.com/in/mateus-alencar-ferreira/" title="Líder & desenvolvedor">🌐</a></td>
  </tr>
</table>
1️⃣ Resumo da Arquitetura do Modelo


A CNN implementada em train_model.py é composta pelas seguintes camadas:

- **Conv2D (16 filtros, 3x3, ReLU):** extrai características básicas das imagens, como bordas e formas
- **MaxPooling2D (2x2):** reduz a dimensionalidade, mantendo as features mais relevantes
- **Conv2D (32 filtros, 3x3, ReLU):** extrai características mais complexas
- **MaxPooling2D (2x2):** nova redução de dimensionalidade
- **Flatten:** transforma o mapa de features em vetor unidimensional
- **Dense (32 neurônios, ReLU):** camada totalmente conectada para classificação
- **Dense (10 neurônios, Softmax):** saída com a probabilidade para cada dígito (0 a 9)

A arquitetura foi mantida simples e leve, com apenas 2 camadas convolucionais, adequada para Edge AI e compatível com os requisitos de tempo do pipeline de CI.
 
---
 
## 2️⃣ Bibliotecas Utilizadas
 
| Biblioteca | Versão | Uso |
|---|---|---|
| TensorFlow | 2.21.0 | Treinamento e conversão do modelo |
| Keras | 3.12+ | Construção da CNN |
| NumPy | 2.2.6 | Manipulação de arrays |
 
---
 
## 3️⃣ Técnica de Otimização do Modelo
 
No arquivo `optimize_model.py` foi aplicada a técnica de **Quantização de Faixa Dinâmica** (Dynamic Range Quantization), utilizando:
 
```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```
 
Essa técnica converte os pesos do modelo de float32 para int8 em tempo de conversão, reduzindo significativamente o tamanho do modelo sem a necessidade de um dataset de calibração. O resultado foi uma redução de **91.1%** no tamanho do modelo, de 1138 KB para 101 KB, tornando-o adequado para execução em dispositivos embarcados e IoT.
 
---
 
## 4️⃣ Resultados Obtidos
 
| Métrica | Valor |
|---|---|
| Acurácia no conjunto de teste | 98.94% |
| Tamanho do modelo original (.h5) | 1138.3 KB |
| Tamanho do modelo otimizado (.tflite) | 101.8 KB |
| Redução de tamanho | 91.1% |
 
---
 
## 5️⃣ Comentários Adicionais
 
**Dificuldades:**
- O pipeline de CI apresentou timeout na primeira execução devido ao tempo de
instalação do TensorFlow somado ao tempo de treinamento. Na segunda execução
o CI rodou com sucesso sem nenhuma alteração nos scripts.

**Decisões técnicas:**
- Optou-se por uma arquitetura com apenas 2 camadas convolucionais para garantir
leveza e compatibilidade com Edge AI.
- O modelo foi treinado por 5 épocas com batch size de 64, equilibrando
velocidade e acurácia.
- A Quantização de Faixa Dinâmica foi escolhida por ser simples de aplicar
e eficaz na redução do tamanho do modelo.

**Aprendizados:**
- Compreensão do fluxo completo de treinamento → salvamento → conversão → otimização para Edge AI.
- Importância de equilibrar acurácia e eficiência em ambientes com restrições de recursos computacionais.


## 📄 Licença

Este projeto está sob a licença do discente Mateus Alencar ferreira, do Processo Seletivo Intensivo Maker 2ª Região - Veja o arquivo <a href="https://github.com/ferreiramateusalencar/PNAAT-processoseletivoIA/blob/main/LICENSE">License.md<a/>

<img width=100% src="https://capsule-render.vercel.app/api?type=waving&color=00bfbf&height=120&section=footer"/>