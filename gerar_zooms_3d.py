"""Gera os 4 modelos 3D de zoom (HTML autonomo, Plotly) -- recortes de detalhe
do sistema intrusivo de Taio: (A) Bela Vista, (B) Pechincha, (C) Dominio norte e
(D) Dominio sul. Cada recorte e um bloco 3D proprio (relevo real do
drone + empilhamento sedimentar + soleira/dique + lineamentos + pontos de campo
+ pocos), com troca de textura (geologia / satelite / hipsometria) e exagero
vertical ajustavel. Os recortes (ZOOMS) sao retangulos UTM 22S (EPSG:31982),
facil de editar.

Uso:  python gerar_zooms_3d.py      ->  zooms_3d_taio.html
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from rasterio import features
from rasterio.transform import from_bounds
from scipy.interpolate import griddata
from shapely.geometry import box

AQUI = Path(__file__).parent
MESTRADO = AQUI.parent.parent
sys.path.insert(0, str(MESTRADO / "4_GIT_TPS" / "mod3d"))
import gerar_visualizador_3d as v  # noqa: E402  (constantes, paleta e construtores de solido do modelo 3D)

BANCO = MESTRADO / "2_Banco_de_Dados"
PONTOS_GPKG = v.PONTOS_CAMPO_GPKG
LINEAMENTOS_GPKG = BANCO / "Unificação" / "lineamentos_direcoes.gpkg"
POCOS_GPKG = BANCO / "Unificação" / "pocos_regionais.gpkg"
LOGO_PATH = AQUI / "assets" / "logo_taio.png"
OUT_HTML = AQUI / "zooms_3d_taio.html"

# retangulos (xmin, ymin, xmax, ymax) em UTM 22S -- AJUSTAR conforme os recortes desejados
ZOOMS = {
    "A": dict(nome="Bela Vista", bbox=(589600, 7006200, 595800, 7015200)),
    "B": dict(nome="Pechincha", bbox=(586000, 7008900, 594200, 7014900)),
    "C": dict(nome="Domínio norte", bbox=(584000, 7015200, 588300, 7018400)),
    "D": dict(nome="Domínio sul", bbox=(582500, 6999700, 590100, 7009300)),
}
N_GRADE = 120          # vertices por eixo maior do relevo
RES_RASTER_M = 12.0    # resolucao do raster geologico (m)
BASE_DIQUE_Z = -350.0  # raiz do dique (m)
ELEV_LINEAMENTO = 14.0
ELEV_PONTO = 22.0
COR_LINEAMENTO = {"Positivo": "#2E7D5B", "Negativo": "#E8A33D"}
PRIORIDADE = ["Teresina", "Serra Alta", "Irati", "Palermo", "Rio Bonito", "Depósito quaternário", "Soleira", "Dique"]
COR_FORMACAO = dict(zip(v.ORDEM_FORMACOES, v.CORES_FORMACOES))


def hex_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def grade_relevo(xyz, bbox, n):
    x0, y0, x1, y1 = bbox
    dx, dy = x1 - x0, y1 - y0
    nx, ny = (n, max(int(n * dy / dx), 20)) if dx >= dy else (max(int(n * dx / dy), 20), n)
    xs, ys = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
    gx, gy = np.meshgrid(xs, ys)
    gz = griddata(xyz[:, :2], xyz[:, 2], (gx, gy), method="linear")
    gz = np.where(np.isnan(gz), griddata(xyz[:, :2], xyz[:, 2], (gx, gy), method="nearest"), gz)
    return gx, gy, gz


def triangulos_grade(ny, nx):
    idx = np.arange(ny * nx).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    return np.concatenate([a, b]), np.concatenate([b, d]), np.concatenate([c, c])


def raster_formacao(lito, bbox, gx, gy):
    """Formacao (indice em PRIORIDADE) em cada vertice da grade -- poligonos queimados na ordem de
    prioridade (dique/soleira por cima), resolucao RES_RASTER_M."""
    x0, y0, x1, y1 = bbox
    w, h = int((x1 - x0) / RES_RASTER_M), int((y1 - y0) / RES_RASTER_M)
    tr = from_bounds(x0, y0, x1, y1, w, h)
    shapes = []
    for k, nome in enumerate(PRIORIDADE):
        for g in lito.loc[lito["formacao"] == nome, "geometry"]:
            if not g.is_empty and g.intersects(box(*bbox)):
                shapes.append((g, k + 1))
    r = features.rasterize(shapes, out_shape=(h, w), transform=tr, fill=0, dtype="uint8")
    col = np.clip(((gx - x0) / (x1 - x0) * (w - 1)).astype(int), 0, w - 1)
    row = np.clip(((y1 - gy) / (y1 - y0) * (h - 1)).astype(int), 0, h - 1)
    return r[row, col]


def mesh_xyz(m, **kw):
    vert, faces = m
    return dict(x=np.round(vert[:, 0], 1), y=np.round(vert[:, 1], 1), z=np.round(vert[:, 2], 1),
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], **kw)


def montar_zoom(chave, cfg, xyz, elevacao, lito, sat, sat_bbox, lineas, pontos, pocos):
    bbox = cfg["bbox"]
    x0, y0, x1, y1 = bbox
    gx, gy, gz = grade_relevo(xyz, bbox, N_GRADE)
    ny, nx = gx.shape
    ti, tj, tk = triangulos_grade(ny, nx)
    base_xy = dict(x=np.round(gx.ravel(), 1), y=np.round(gy.ravel(), 1), z=np.round(gz.ravel(), 1), i=ti, j=tj, k=tk)
    fig = go.Figure()

    # --- relevo em 3 texturas (so uma visivel por vez) ---
    form = raster_formacao(lito, bbox, gx, gy)
    paleta = np.array([[60, 60, 70]] + [hex_rgb(COR_FORMACAO[n]) for n in PRIORIDADE], dtype=np.uint8)
    cor_geo = paleta[form].reshape(-1, 3)
    sr, sg, sb = v.amostrar_satelite_rgb(sat, gx, gy, *sat_bbox)
    cor_sat = np.column_stack([sr.ravel(), sg.ravel(), sb.ravel()])
    zn = (gz.ravel() - gz.min()) / max(np.ptp(gz), 1.0)
    paleta_h = np.array([hex_rgb(c) for c in v.CORES_HIPSOMETRICAS])
    pos = zn * (len(paleta_h) - 1)
    lo = np.floor(pos).astype(int).clip(0, len(paleta_h) - 2)
    fr = (pos - lo)[:, None]
    cor_hip = (paleta_h[lo] * (1 - fr) + paleta_h[lo + 1] * fr).astype(np.uint8)
    for nome, cores, vis in (("tex_geologia", cor_geo, True), ("tex_satelite", cor_sat, False), ("tex_hipsometria", cor_hip, False)):
        fig.add_trace(go.Mesh3d(**base_xy, vertexcolor=cores, name=nome, visible=vis, showlegend=False, hoverinfo="skip",
                                lighting=dict(ambient=0.7, diffuse=0.6, specular=0.05), flatshading=False, meta=nome))

    # --- empilhamento (paredes do bloco) ---
    tilt = v.Z_REF_TILT + v.TREND_A * (gx - v.TREND_X0) + v.TREND_B * (gy - v.TREND_Y0)
    contatos = [gz if m == 0 else np.minimum(tilt - v.PROFUNDIDADE_CAMADAS[m], gz) for m in range(len(v.PROFUNDIDADE_CAMADAS))]
    # "erase": dentro da soleira as camadas acima da base do corpo somem (a soleira ocupa terreno -> base);
    # dentro do dique as camadas somem por completo (o dique atravessa todo o pacote).
    m_sill = (form == PRIORIDADE.index("Soleira") + 1).reshape(gz.shape)
    m_dique = (form == PRIORIDADE.index("Dique") + 1).reshape(gz.shape)
    base_sill_g = np.minimum(contatos[1], gz - v.ESPESSURA_MINIMA_SILL)
    contatos = [np.where(m_dique, gz, np.where(m_sill, np.minimum(c, base_sill_g), c)) for c in contatos]
    for k, nome in enumerate(v.NOMES_CAMADAS):
        ri = np.unique(np.append(np.arange(0, ny, 3), ny - 1)); ci = np.unique(np.append(np.arange(0, nx, 3), nx - 1))
        sub = np.ix_(ri, ci)  # paredes com malha mais grossa (tiras largas, sem serrilhado)
        px, py, pz, pi, pj, pk = v.paredes_caixa(gx[sub], gy[sub], contatos[k][sub], contatos[k + 1][sub])
        fig.add_trace(go.Mesh3d(x=np.round(px, 1), y=np.round(py, 1), z=np.round(pz, 1), i=pi, j=pj, k=pk,
                                color=v.CORES_CAMADAS[k], flatshading=False, name=nome, legendgroup=nome, showlegend=True,
                                hoverinfo="name", lighting=dict(ambient=0.95, diffuse=0.1, specular=0.0, roughness=1.0)))
    zmin_bloco = float(np.nanmin(contatos[-1]))

    # --- soleira e dique (solidos) ---
    janela = box(*bbox)
    for nome, cor, kw in (("Soleira", v.COR_SILL, dict(base_fn=lambda x, y: v.elevacao_serra_alta(x, y, elevacao),
                                                      espessura_minima=v.ESPESSURA_MINIMA_SILL)),
                          ("Dique", v.COR_DIQUE, dict(base_absoluta=BASE_DIQUE_Z))):
        mostrou = False
        for g in lito.loc[lito["formacao"] == nome, "geometry"]:
            inter = g.intersection(janela)
            partes = [inter] if inter.geom_type == "Polygon" else list(getattr(inter, "geoms", []))
            for p in partes:
                if p.geom_type != "Polygon" or p.area < 400:
                    continue
                vert, faces = v.construir_solido(p, elevacao, **kw)
                if len(faces) == 0:
                    continue
                fig.add_trace(go.Mesh3d(**mesh_xyz((vert, faces)), color=cor, flatshading=True, name=nome, legendgroup=nome,
                                        showlegend=not mostrou, hoverinfo="name", lighting=dict(ambient=0.7, diffuse=0.55)))
                mostrou = True

    # --- lineamentos / campo / pocos (grupos ligaveis) ---
    for tipo, cor in COR_LINEAMENTO.items():
        xs, ys, zs = [], [], []
        for g in lineas.loc[lineas["tipo"] == tipo, "geometry"]:
            inter = g.intersection(janela)
            for seg in ([inter] if inter.geom_type == "LineString" else list(getattr(inter, "geoms", []))):
                if seg.geom_type != "LineString" or seg.length < 5:
                    continue
                pts = [(seg.interpolate(d).x, seg.interpolate(d).y) for d in np.linspace(0, seg.length, max(int(seg.length // 60), 2))]
                xs += [p[0] for p in pts] + [None]; ys += [p[1] for p in pts] + [None]
                zs += [elevacao(*p) + ELEV_LINEAMENTO for p in pts] + [None]
        if xs:
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=cor, width=5), name=f"Lineamento {tipo.lower()}",
                                       legendgroup="lin", meta="lineamentos", hoverinfo="name"))
    dentro = pontos[pontos.geometry.within(janela)]
    if len(dentro):
        fig.add_trace(go.Scatter3d(
            x=dentro.X, y=dentro.Y, z=[elevacao(x, y) + ELEV_PONTO for x, y in zip(dentro.X, dentro.Y)], mode="markers",
            marker=dict(size=4, color=[v.CORES_LITOLOGIA_CAMPO.get(l, v.COR_LITOLOGIA_PADRAO) for l in dentro["litologia_padronizada"]],
                        line=dict(width=1, color="#071729")),
            text=dentro["ponto_id"].astype(str), hoverinfo="text", name="Pontos de campo", meta="campo"))
    pd_ = pocos[pocos.geometry.within(janela)]
    if len(pd_):
        fig.add_trace(go.Scatter3d(
            x=pd_.geometry.x, y=pd_.geometry.y, z=[elevacao(p.x, p.y) + 5 for p in pd_.geometry], mode="markers+text",
            marker=dict(size=6, symbol="diamond", color="#FFFFFF"), text=pd_["nome"], hoverinfo="text", name="Poços", meta="pocos"))

    dx, dy = x1 - x0, y1 - y0
    zmax = float(np.nanmax(gz)) + 60
    zmin = max(zmin_bloco, -450.0)
    fig.update_layout(
        paper_bgcolor=v.MARCA_NAVY, font=dict(family=v.MARCA_FONTE, color=v.MARCA_CINZA_CLARO), margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(bgcolor="rgba(12,41,69,0.8)", bordercolor=v.MARCA_ROXO, borderwidth=1, x=0.01, y=0.99),
        scene=dict(
            xaxis=dict(title="X (UTM)", range=[x0 - 0.03 * dx, x1 + 0.03 * dx], backgroundcolor="#0C2945", gridcolor="#2A4663", color=v.MARCA_CINZA_CLARO),
            yaxis=dict(title="Y (UTM)", range=[y0 - 0.03 * dy, y1 + 0.03 * dy], backgroundcolor="#0C2945", gridcolor="#2A4663", color=v.MARCA_CINZA_CLARO),
            zaxis=dict(title="Z (m)", range=[zmin - 40, zmax], backgroundcolor="#0C2945", gridcolor="#2A4663", color=v.MARCA_CINZA_CLARO),
            aspectmode="manual", aspectratio=dict(x=1.06, y=1.06 * dy / dx, z=(zmax - zmin + 40) * 3.0 / dx),
            camera=dict(eye=dict(x=-1.25, y=-1.45, z=0.95)),
        ),
    )
    info = dict(nome=cfg["nome"], dx=dx, dy=dy, dz=zmax - zmin, area_km2=dx * dy / 1e6,
                n_pontos=int(len(dentro)), n_pocos=int(len(pd_)))
    return fig.to_json(), info


def main():
    print("Carregando dados...")
    xyz = np.load(v.TOPO_NPY)
    elevacao = v.montar_elevador(xyz)
    sat_bbox = (xyz[:, 0].min(), xyz[:, 1].min(), xyz[:, 0].max(), xyz[:, 1].max())
    sat = v.obter_satelite_utm(*sat_bbox)
    lito = gpd.read_file(v.LITOLOGIA_ATUALIZADA).to_crs(31982)
    lineas = gpd.read_file(LINEAMENTOS_GPKG).to_crs(31982)
    pontos = gpd.read_file(PONTOS_GPKG).to_crs(31982)
    pocos = gpd.read_file(POCOS_GPKG).to_crs(31982)
    figs, infos = {}, {}
    for chave, cfg in ZOOMS.items():
        print(f"  zoom {chave}: {cfg['nome']}")
        figs[chave], infos[chave] = montar_zoom(chave, cfg, xyz, elevacao, lito, sat, sat_bbox, lineas, pontos, pocos)
        print(f"    {infos[chave]['area_km2']:.1f} km², {infos[chave]['n_pontos']} pontos de campo, {infos[chave]['n_pocos']} poços")

    import base64
    logo = base64.b64encode(LOGO_PATH.read_bytes()).decode() if LOGO_PATH.exists() else ""
    abas = "".join(f'<button class="aba{" ativo" if i == 0 else ""}" data-z="{c}">({c}) {ZOOMS[c]["nome"]}</button>' for i, c in enumerate(ZOOMS))
    html = f"""<!DOCTYPE html>
<html lang="pt-br"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Zooms 3D — Taió Plumbing System</title>
<link rel="icon" type="image/png" href="assets/favicon.png?v=2"><link rel="shortcut icon" href="assets/favicon.ico?v=2">
<link rel="apple-touch-icon" href="assets/apple-touch-icon.png?v=2">
<style>
 :root {{ --bg:{v.MARCA_NAVY}; --painel:{v.MARCA_ROXO_ESCURO}; --txt:{v.MARCA_CINZA_CLARO}; --ac:{v.MARCA_ROXO}; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; background:var(--bg); color:var(--txt); font-family:{v.MARCA_FONTE}; overflow:hidden; }}
 #barra {{ display:flex; flex-wrap:wrap; gap:6px 14px; align-items:center; padding:8px 14px; background:var(--painel); border-bottom:2px solid var(--ac); }}
 #barra h1 {{ font-size:15px; margin:0 8px 0 0; }} #barra h1 b {{ color:var(--ac); }}
 .grupo {{ display:flex; gap:5px; align-items:center; font-size:12px; }}
 .grupo label {{ opacity:.7; text-transform:uppercase; font-size:10px; letter-spacing:.06em; }}
 button, select {{ background:var(--bg); color:var(--txt); border:1px solid var(--ac); border-radius:5px; padding:4px 9px; font:inherit; font-size:12px; cursor:pointer; }}
 button.ativo {{ background:var(--ac); color:var(--bg); }}
 #info {{ margin-left:auto; font-size:11px; opacity:.75; }}
 #logo {{ position:fixed; top:8px; right:12px; width:46px; height:46px; border-radius:50%; border:2px solid var(--ac); z-index:5; }}
 .plot {{ width:100%; display:none; }} .plot.ativo {{ display:block; }}
</style></head><body>
<img id="logo" src="data:image/png;base64,{logo}">
<div id="barra"><h1><b>Zooms 3D</b> — Taió</h1>
 <div class="grupo"><label>Recorte</label>{abas}</div>
 <div class="grupo"><label>Textura</label><button class="tex ativo" data-t="tex_geologia">Geologia</button><button class="tex" data-t="tex_satelite">Satélite</button><button class="tex" data-t="tex_hipsometria">Hipsometria</button></div>
 <div class="grupo"><label>Exagero vertical</label><select id="exag"><option value="1">1×</option><option value="2">2×</option><option value="3" selected>3×</option><option value="5">5×</option><option value="8">8×</option></select></div>
 <div class="grupo"><label>Camadas</label><button class="cam ativo" data-m="lineamentos">Lineamentos</button><button class="cam ativo" data-m="campo">Campo</button><button class="cam ativo" data-m="pocos">Poços</button></div>
 <div id="info"></div></div>
{"".join(f'<div class="plot{" ativo" if i == 0 else ""}" id="plot-{c}"></div>' for i, c in enumerate(ZOOMS))}
<script>{get_plotlyjs()}</script>
<script>
var FIGS = {{{",".join(f'"{c}": {figs[c]}' for c in ZOOMS)}}};
var INFOS = {json.dumps(infos, ensure_ascii=False)};
var criados = {{}}, atual = 'A', textura = 'tex_geologia', EXAG = 3, camadasOn = {{lineamentos: true, campo: true, pocos: true}};
function altura() {{ return window.innerHeight - document.getElementById('barra').offsetHeight; }}
function aplicar(c) {{
  var gd = document.getElementById('plot-' + c), fig = FIGS[c];
  var vis = fig.data.map(function(t) {{
    var m = t.meta;
    if (m && m.indexOf('tex_') === 0) return m === textura;
    if (m && camadasOn.hasOwnProperty(m)) return camadasOn[m];
    return true;
  }});
  Plotly.restyle(gd, {{visible: vis}});
  var s = fig.layout.scene, ar = s.aspectratio;
  Plotly.relayout(gd, {{'scene.aspectratio': {{x: ar.x, y: ar.y, z: ar.z * EXAG / 3}}}});
}}
function mostrar(c) {{
  atual = c;
  document.querySelectorAll('.plot').forEach(function(e) {{ e.classList.toggle('ativo', e.id === 'plot-' + c); }});
  document.querySelectorAll('.aba').forEach(function(b) {{ b.classList.toggle('ativo', b.dataset.z === c); }});
  var gd = document.getElementById('plot-' + c); gd.style.height = altura() + 'px';
  if (!criados[c]) {{
    Plotly.newPlot(gd, FIGS[c].data, FIGS[c].layout, {{responsive: true, displaylogo: false}}); criados[c] = true;
  }} else {{ Plotly.Plots.resize(gd); }}
  aplicar(c);
  var i = INFOS[c];
  document.getElementById('info').textContent = i.nome + ' · ' + i.area_km2.toFixed(1) + ' km² · ' + i.n_pontos + ' pontos de campo · ' + i.n_pocos + ' poços';
}}
document.querySelectorAll('.aba').forEach(function(b) {{ b.onclick = function() {{ mostrar(b.dataset.z); }}; }});
document.querySelectorAll('.tex').forEach(function(b) {{ b.onclick = function() {{
  textura = b.dataset.t; document.querySelectorAll('.tex').forEach(function(x) {{ x.classList.toggle('ativo', x === b); }}); aplicar(atual); }}; }});
document.querySelectorAll('.cam').forEach(function(b) {{ b.onclick = function() {{
  camadasOn[b.dataset.m] = !camadasOn[b.dataset.m]; b.classList.toggle('ativo', camadasOn[b.dataset.m]); aplicar(atual); }}; }});
document.getElementById('exag').onchange = function() {{ EXAG = parseFloat(this.value); aplicar(atual); }};
window.addEventListener('resize', function() {{ var gd = document.getElementById('plot-' + atual); gd.style.height = altura() + 'px'; Plotly.Plots.resize(gd); }});
mostrar('A');
</script></body></html>"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Salvo em: {OUT_HTML} ({OUT_HTML.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
