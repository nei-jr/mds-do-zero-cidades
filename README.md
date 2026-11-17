# MDS clássico do zero: redesenhando o mapa a partir das distâncias

Se eu tiver só a tabela de distâncias entre 14 cidades, sem nenhuma coordenada, consigo redesenhar o mapa?

Neste notebook implemento o MDS clássico (Multidimensional Scaling) em PyTorch, sem usar `sklearn.manifold.MDS`, e testo isso com cidades de SP, PR, SC, RJ, MG, MS e DF. As coordenadas reais servem só para calcular as distâncias e conferir o resultado no final.

![Mapa real e mapa reconstruído pelo MDS](img/mapa_real_vs_mds.png)

## O que tem no notebook

1. Matriz de distâncias pela fórmula de haversine, calculada para todos os pares de uma vez usando broadcasting.
2. MDS clássico: dupla centralização $B = -\tfrac12 H D^{(2)} H$ e decomposição espectral $Y = V_d\Lambda_d^{1/2}$. A dupla centralização é conferida entrada a entrada com a fórmula das médias.
3. Comparação com o mapa real. O erro médio nas distâncias fica em 0,11 km (máximo de 0,83 km), com distâncias de até 1314 km.
4. Alinhamento por Procrustes ortogonal, mostrando que a diferença entre o mapa do MDS e o real é só uma rotação.
5. Leitura dos autovalores de $B$, onde aparece a curvatura da Terra.
6. Comparação entre arco e corda.

![MDS alinhado sobre o mapa real](img/mds_alinhado.png)

## A parte que eu achei mais interessante

Se as cidades estivessem num plano, $B$ teria exatamente 2 autovalores positivos e o resto seria zero. Não é o que acontece:

- $\lambda_1$ e $\lambda_2$ são da ordem de $10^6$ e formam o mapa.
- $\lambda_3 \approx 674$ é positivo e cerca de 1000 vezes menor. É a profundidade da calota esférica que contém as cidades.
- Aparecem autovalores negativos (o maior chega a $-1400$). Isso seria impossível se as distâncias fossem euclidianas. Eles aparecem porque a haversine mede o arco sobre a superfície, que é um pouco mais longo que a reta.

Para testar essa explicação, troquei o arco pela corda (a linha reta que atravessa a Terra), que é uma distância euclidiana de verdade:

![Espectro de B com arco e com corda](img/espectro_arco_vs_corda.png)

| | Arco (haversine) | Corda |
|---|---|---|
| Autovalores positivos | 7 | 3 |
| Autovalores negativos | 6 | 0 |
| Zeros numéricos | 1 | 11 |
| $\lambda_3$ | ≈ 674 | ≈ 2074 |

Com a corda, os negativos somem e $B$ fica com posto exatamente 3: só pela tabela de distâncias, o MDS descobre que os pontos estão em $\mathbb{R}^3$. Os 3 autovalores batem com os autovalores de $X^\top X$ calculados a partir das posições 3D reais das cidades, e o MDS em 3D reproduz as cordas com erro da ordem de $10^{-13}$ km.

Arco e corda diferem em no máximo 2,3 km nesta tabela. Essa diferença de 0,2% já basta para a matriz deixar de ser euclidiana, e o espectro mostra isso.

Para separar o que é zero de verdade do que é ruído de arredondamento, usei a tolerância $N\,\varepsilon\,|\lambda_1|$, o mesmo critério do `numpy.linalg.matrix_rank`.

## Como rodar

```bash
pip install torch matplotlib jupyter
jupyter notebook mds_cidades.ipynb
```

Testado com Python 3.13, PyTorch 2.15 e Matplotlib 3.11. Tudo roda em CPU em poucos segundos.

## Referência

Murphy, K. P. *Probabilistic Machine Learning: An Introduction*. MIT Press, 2022. Seção 20.4.4 (Multidimensional scaling).

## Ligação com o Isomap

O Isomap é este mesmo procedimento com outra medida de distância: no lugar da haversine, entra uma distância geodésica estimada num grafo de vizinhos. Aqui arco e reta quase não diferem, porque a região é pequena perto do raio da Terra. No Swiss roll é o contrário, e é aí que a diferença importa.
