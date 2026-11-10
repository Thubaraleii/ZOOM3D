# Zoom 3D — Taió

**Versão 1.0** · Projeto *Taió Plumbing System* (Mestrado em Geologia, UFSC)

🔗 **Acesso:** <https://thubaraleii.github.io/ZOOM3D/zooms_3d_taio.html>

Cinco modelos 3D de detalhe (Plotly, HTML autônomo) do sistema intrusivo de Taió, cada um como **bloco sólido** com relevo,
empilhamento sedimentar, soleiras e diques em volume, e ferramenta de corte.

Inclui também a página **Recorte regional (furos)** — ver abaixo.

## Recortes

| Aba | Recorte |
|---|---|
| A | Bela Vista |
| B | Pechincha |
| C | Domínio norte |
| D | Domínio sul |
| E | Horst (esquemático) |

Os retângulos (UTM 22S, SIRGAS 2000) estão no dicionário `ZOOMS` de `gerar_zooms_3d.py`.

## Funcionalidades

- **Textura:** geologia (polígonos exatos colados no relevo), satélite ou hipsometria.
- **Corte** em X (Leste–Oeste) ou Y (Norte–Sul), com controle deslizante e **Inverter lado**. O corte recorta a malha por um plano e calcula a **face do corte** no navegador (empilhamento, soleira e dique); a caixa e a escala permanecem fixas.
- **Camadas:** pontos de campo, geoquímica (Alto/Baixo-Ti), poços, e estruturas — falha CPRM, fratura CPRM, falha de campo (traço de rumo) e lineamentos positivo/negativo, cada um com seu botão.
- **Exagero vertical** de 1× a 8× (padrão 3×).

## Aba E — Horst (modelo esquemático)

Bloco de todo o domínio de Taió (≈ 23 × 35 km) com o **horst** definido em `horst.shp` (linha “Alto”: três falhas que o limitam; linhas “baixo”: extremidades) e **grabens subordinados** nos dois flancos.

- **Zonas rígidas:** horst, grabens (faixa de 3 km a cada lado do horst) e restante do bloco; relevo, decalques geológicos, soleiras, diques, paredes e camadas vetoriais são recortados por zona, de modo que as falhas são degraus verticais exatos.
- **Basculamento:** o rejeito do horst varia linearmente de ~8 % do máximo na borda norte (linha “baixo”) a 100 % na quina sul; os grabens afundam até 35 % do rejeito do horst, com suavização perto da extremidade sul.
- **Controles:** controle deslizante de **rejeito** (0–300 m) e botão **Subir / descer** (única animação); camadas **Planos de falha** (fitas verticais do topo à base), **Traços** (falhas do horst, limites “baixo” e falhas subordinadas) e **Rótulos**.
- **Paredes de falha:** nas bordas do horst (e nas falhas dos grabens) o bloco mais alto exibe a coluna completa — cada unidade e a base — de modo que o degrau exposto mostra as camadas.
- **Corte** X/Y com face de corte que acompanha os degraus nas falhas; laje cinza “Base (não modelada)” de 300 m abaixo do pacote sedimentar, só para que as falhas desçam além do modelo estratigráfico.
- **Parâmetros** (`gerar_zooms_3d.py`): `LARGURA_GRABEN`, `T_MAX_HORST`, `A_HORST`, `G_GRABEN`, `EXTRA_BASE`.
- O rejeito é **ilustrativo**: com rejeito 0 o bloco é o modelo calibrado nos dados; valores maiores são uma representação esquemática, não uma medida.

## Dados e métodos

- Malha de relevo de ~200 vértices no eixo maior, a partir do MDE de drone; geologia triangulada com os nós da malha, sem sobreposição.
- Soleira e dique apagam o sedimentar: as camadas acima da base do corpo não existem dentro do polígono, de modo que não há superfícies sobrepostas.
- Chão do bloco na base do empilhamento (854 m de profundidade cumulativa no modelo).

## Recorte regional (furos) — `regional_furos_3d.html`

Bloco regional que cobre o raio dos **10 poços ANP mais próximos de Taió** e os poços do SAG; as dimensões do bloco e o raio são calculados na geração a partir da distância do décimo poço.

- Relevo (AWS Terrain Tiles) em hipsometria ou geologia SC (CPRM); falhas, fraturas, diques e lineamentos; área de estudo e círculo do raio dos poços.
- Poços como hastes da boca ao fundo (profundidade total) e tabela lateral com distância, profundidade e cota do fundo.
- **Superfícies das unidades** indicativas, obtidas por extrapolação do modelo regional de Taió. A tabela de poços da ANP **não contém topos de formação nem litologia**, portanto as superfícies não são amarradas aos poços.
- **Calibração:** as superfícies são ajustadas aos topos de `2_Banco_de_Dados/dados_base/topos_pocos.csv` (`poco, formacao, prof_topo_m`; formações Irati, Palermo e Rio Bonito, e a linha `Rio Bonito (base)` para a base do modelo) como tendência regional + resíduo interpolado. Teresina e Serra Alta não têm topo separado na fonte e não são calibradas. Interfaces são forçadas a manter a ordem estratigráfica (sem espessura negativa). Longe dos poços as superfícies são extrapoladas e devem ser lidas como indicativas.

## Idioma e navegação

- **Idioma:** o botão **EN / PT**, ao lado do logotipo, alterna a interface entre português e inglês em tempo real (menus, legendas, dicas, popups, rótulos e textos 3D). O padrão é **sempre o português**; o inglês vale na página aberta com `?lang=en`, parâmetro que o botão e os links entre as páginas repassam.
- **Hub:** o botão **← Hub** volta à página inicial do projeto, mantendo o idioma.
- **Tradução:** feita no navegador por dicionário (`tps_i18n_dict.py`, embutido na página por `tps_i18n.py` ao gerar o HTML). Dados de origem em português (por exemplo, descrições de campo e nomes próprios de unidades) permanecem no idioma original.

## Uso

Abra `zooms_3d_taio.html` (e `regional_furos_3d.html`) no navegador ou acesse as versões publicadas. Arraste para orbitar e role para ampliar.

## Regenerar os produtos

O repositório contém o HTML publicado e o script gerador. Para regenerar é necessário o conjunto de dados do projeto
(modelo digital de elevação, litologia, pontos de campo, lineamentos, poços), que **não é distribuído** neste repositório.

```bash
python gerar_zooms_3d.py  # e gerar_regional_furos.py
# a aba E também lê 2_Banco_de_Dados/dados_base/horst.shp
```

Dependências Python: `numpy`, `pandas`, `geopandas`, `shapely`, `pyproj`, `rasterio`, `scipy`, `plotly`, `contextily`.

Requer a biblioteca `gerar_visualizador_3d.py` (Modelo 3D) e, para o recorte regional, o cache de tiles do panorama.

## Estrutura

```
zooms_3d_taio.html           4 recortes
regional_furos_3d.html       recorte regional dos furos
gerar_zooms_3d.py            gerador dos recortes
gerar_regional_furos.py      gerador do recorte regional
assets/                      logotipo e ícones
```

## Referências

1. Milani, E. J.; Melo, J. H. G.; Souza, P. A.; Fernandes, L. A.; França, A. B. (2007). Bacia do Paraná. *Boletim de Geociências da Petrobras*, 15(2), 265–287.
2. Schneider, R. L.; Mühlmann, H.; Tommasi, E.; Medeiros, R. A.; Daemon, R. F.; Nogueira, A. A. (1974). Revisão estratigráfica da Bacia do Paraná. In: *Anais do 28º Congresso Brasileiro de Geologia*, Porto Alegre, SBG, 1, 41–65.
3. Frank, H. T.; Gomes, M. E. B.; Formoso, M. L. L. (2009). Review of the areal extent and the volume of the Serra Geral Formation, Paraná Basin, South America. *Pesquisas em Geociências*, 36(1), 49–57.
4. Peate, D. W. (1997). The Paraná–Etendeka Province. In: Mahoney, J. J.; Coffin, M. F. (eds.), *Large Igneous Provinces*. AGU Geophysical Monograph 100, 217–245.
5. Agafonkin, V. **Leaflet** — an open-source JavaScript library for interactive maps. <https://leafletjs.com>
6. Plotly Technologies Inc. (2015). **Collaborative data science** (Plotly.js). Montréal. <https://plot.ly>
7. Mapzen / Amazon Web Services. **Terrain Tiles** (formato Terrarium). <https://registry.opendata.aws/terrain-tiles/>
8. OpenStreetMap contributors. **OpenStreetMap**. <https://www.openstreetmap.org/copyright>
9. CPRM/SGB — Serviço Geológico do Brasil. **Geologia e Recursos Minerais do Estado de Santa Catarina — Mapa Geológico 1:500.000 (SIG SC)**. <https://geosgb.sgb.gov.br>
10. ANP — Agência Nacional do Petróleo, Gás Natural e Biocombustíveis. **Dados abertos — poços**. <https://www.gov.br/anp>
11. Harris, C. R. et al. (2020). Array programming with NumPy. *Nature*, 585, 357–362.
12. Virtanen, P. et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261–272.
13. Jordahl, K. et al. **GeoPandas**. Zenodo. <https://geopandas.org>; Gillies, S. et al. **Shapely** e **Rasterio**. <https://github.com/shapely/shapely>, <https://github.com/rasterio/rasterio>.

## Licença e uso

- **Código** (scripts de geração e JavaScript das páginas): licença **MIT** — ver [`LICENSE`](LICENSE).
- **Produtos gerados** (HTML, figuras, tabelas derivadas): **CC BY 4.0** — uso, adaptação e redistribuição permitidos com atribuição ao autor.
- **Dados de terceiros** incorporados mantêm as licenças e os termos das fontes originais (tabela em *Créditos e fontes de dados*).
- Os modelos são **interpretativos** e dependem da qualidade e da densidade dos dados de entrada; **não substituem** estudo geológico, hidrogeológico ou geotécnico de detalhe, nem devem ser usados como base única para decisões de engenharia, perfuração ou licenciamento.
- Fornecido “como está”, sem garantias de qualquer tipo.

## Créditos e fontes de dados

| Fonte | Uso | Termos |
|---|---|---|
| Levantamento de campo do autor (pontos, estruturas, geoquímica, MDE de drone) | Dados primários da área de Taió | CC BY 4.0 (autor) |
| CPRM/SGB — Mapa Geológico do Estado de Santa Catarina 1:500.000 (SIG SC) | Geologia regional, falhas, fraturas e diques regionais | Dados abertos CPRM/SGB — citar a fonte |
| ANP — Tabela de poços (dados abertos) | Poços de petróleo/gás (posição e profundidade total) | Dados abertos ANP — citar a fonte |
| Cadastro de poços do Sistema Aquífero Guarani (SAG) | Poços do SAG (posição e profundidade), arquivo KML do projeto | Citar a fonte original do cadastro |
| OpenStreetMap contributors | Rios, estradas e localidades | ODbL 1.0 — © OpenStreetMap contributors |
| Esri World Imagery | Imagem de satélite (tiles e textura) | Termos da Esri — uso com atribuição |
| CARTO / OpenTopoMap | Mapas base do webmap | Atribuição nos próprios mapas (OSM, SRTM, OpenTopoMap CC-BY-SA) |
| AWS Terrain Tiles (formato Terrarium) | Relevo e batimetria regionais (derivados de SRTM, GEBCO e outras fontes) | Atribuição conforme o provedor dos tiles |
| Plotly.js (MIT), Leaflet (BSD-2-Clause) | Bibliotecas de visualização embutidas/carregadas nas páginas | Licenças originais |

## Como citar

> JESUS, A. H. de. **Zoom 3D — Taió**, versão 1.0. 2026. Projeto *Taió Plumbing System* — Mestrado em Geologia, UFSC. Disponível em: <https://thubaraleii.github.io/ZOOM3D/zooms_3d_taio.html>.

## Autoria

**Afonso Henrique de Jesus** — Mestrado em Geologia, Universidade Federal de Santa Catarina (UFSC).
Projeto: *Architecture and Geochemical Evolution of the Taió Intrusive Plumbing System* (Taió, SC — Bacia do Paraná).
