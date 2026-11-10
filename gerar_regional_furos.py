"""Recorte regional dos FUROS (HTML autonomo, Plotly): bloco ("cubao") que cobre o raio dos 10 pocos ANP mais
proximos de Taio + os 2 pocos SAG, com relevo (hipsometria ou geologia SC/CPRM), lineamentos/falhas/fraturas,
pocos (haste da boca ao fundo, profundidade total) e superficies INDICATIVAS das unidades.

IMPORTANTE (limite dos dados): a tabela de pocos da ANP traz posicao e profundidade total, mas NAO traz topos de
formacao/litologia (campos GEOLOGIA_* vazios). Por isso as superficies das unidades aqui sao a EXTRAPOLACAO do
modelo regional de Taio (plano de mergulho + espessuras), sem calibracao nos pocos. Quando houver topos reais,
preencher `2_Banco_de_Dados/dados_base/topos_pocos.csv` (poco, formacao, prof_topo_m) -- o script passa a calibrar
as superficies nos pocos (tendencia + residuo interpolado).

Uso:  python gerar_regional_furos.py   ->  regional_furos_3d.html
"""
import base64
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from pyproj import Transformer
from rasterio import features
from rasterio.transform import from_bounds
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
from shapely.geometry import box

AQUI = Path(__file__).parent
MESTRADO = AQUI.parent.parent
sys.path.insert(0, str(MESTRADO / "4_GIT_TPS" / "mod3d"))
import gerar_visualizador_3d as v  # noqa: E402  (paleta, constantes do modelo regional de Taio, paredes_caixa)

BANCO = MESTRADO / "2_Banco_de_Dados"
CACHE = MESTRADO / "4_GIT_TPS" / "panorama3d" / "cache"
POCOS_GPKG = BANCO / "Unificação" / "pocos_regionais.gpkg"
LINEAMENTOS_GPKG = BANCO / "Unificação" / "lineamentos_direcoes.gpkg"
GEOLOGIA_REGIONAL_GPKG = BANCO / "dados_base" / "geologia_regional_sc.gpkg"
TOPOS_CSV = BANCO / "dados_base" / "topos_pocos.csv"
LOGO_PATH = AQUI / "assets" / "logo_taio.png"
OUT_HTML = AQUI / "regional_furos_3d.html"

CRS_UTM = "EPSG:31982"
CENTRO_TAIO = (593500.0, 7012000.0)
TAIO_AREA = (582500.0, 6995000.0, 604500.0, 7029000.0)
N_ANP = 10
MARGEM_M = 12000.0
N_GRADE = 240
EXAGERO_PADRAO = 10
AZUL_FALHA, ROSA_FRATURA = "#2563EB", "#D81B73"
COR_UNIDADE = dict(zip(v.NOMES_CAMADAS, v.CORES_CAMADAS))


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)


def amostrar_mosaico(img, ext, gx, gy):
    lon, lat = Transformer.from_crs(CRS_UTM, 4326, always_xy=True).transform(gx, gy)
    mx, my = Transformer.from_crs(4326, 3857, always_xy=True).transform(lon, lat)
    h, w = img.shape[:2]
    col = (mx - ext[0]) / (ext[1] - ext[0]) * w - 0.5
    row = (ext[3] - my) / (ext[3] - ext[2]) * h - 0.5
    return np.stack([map_coordinates(img[:, :, c].astype(float), [row, col], order=1, mode="nearest") for c in range(img.shape[2])], axis=-1)


def triangulos_grade(ny, nx):
    idx = np.arange(ny * nx).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    return np.concatenate([a, b]), np.concatenate([b, d]), np.concatenate([c, c])


def main():
    print("Carregando pocos...")
    pocos = gpd.read_file(POCOS_GPKG).to_crs(CRS_UTM)
    pocos["dist_km"] = np.hypot(pocos.geometry.x - CENTRO_TAIO[0], pocos.geometry.y - CENTRO_TAIO[1]) / 1000
    anp = pocos[pocos["fonte"].str.startswith("ANP")].sort_values("dist_km").head(N_ANP)
    sag = pocos[~pocos["fonte"].str.startswith("ANP")]
    sel = pd.concat([anp, sag]).reset_index(drop=True)
    raio_km = float(anp["dist_km"].max())
    print(f"  {len(anp)} pocos ANP (ate {raio_km:.0f} km de Taio) + {len(sag)} SAG")

    xmin = min(sel.geometry.x.min(), TAIO_AREA[0]) - MARGEM_M
    xmax = max(sel.geometry.x.max(), TAIO_AREA[2]) + MARGEM_M
    ymin = min(sel.geometry.y.min(), TAIO_AREA[1]) - MARGEM_M
    ymax = max(sel.geometry.y.max(), TAIO_AREA[3]) + MARGEM_M
    dx, dy = xmax - xmin, ymax - ymin
    nx, ny = (N_GRADE, max(int(N_GRADE * dy / dx), 20)) if dx >= dy else (max(int(N_GRADE * dx / dy), 20), N_GRADE)
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))
    print(f"  recorte: {dx / 1000:.0f} x {dy / 1000:.0f} km")

    elev_img, elev_ext = np.load(CACHE / "terrarium_z9.npy"), np.load(CACHE / "terrarium_z9_ext.npy")
    rgb = amostrar_mosaico(elev_img, elev_ext, gx, gy)
    gz = np.round(rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768, 0)
    gz = np.maximum(gz, 0.0)
    ti, tj, tk = triangulos_grade(ny, nx)
    base = dict(x=np.round(gx.ravel(), 0), y=np.round(gy.ravel(), 0), z=gz.ravel(), i=ti, j=tj, k=tk)

    def z_em(px, py, extra=0.0):
        r = amostrar_mosaico(elev_img, elev_ext, np.asarray(px, float), np.asarray(py, float))
        return np.maximum(r[..., 0] * 256 + r[..., 1] + r[..., 2] / 256 - 32768, 0.0) + extra

    # ---- cores do relevo: hipsometria e geologia SC (CPRM) ----
    terra = np.array([hex_rgb(c) for c in v.CORES_HIPSOMETRICAS])
    zt = np.clip(gz / 1100.0, 0, 1) * (len(terra) - 1)
    lo = np.floor(zt).astype(int).clip(0, len(terra) - 2)
    fr = (zt - lo)[..., None]
    cor_hip = (terra[lo] * (1 - fr) + terra[lo + 1] * fr).astype(np.uint8).reshape(-1, 3)
    geo = gpd.read_file(GEOLOGIA_REGIONAL_GPKG, layer="litologia")
    res = 500.0
    w_r, h_r = int(dx / res), int(dy / res)
    rast = features.rasterize([(g, k + 1) for k, g in enumerate(geo.geometry)], out_shape=(h_r, w_r),
                              transform=from_bounds(xmin, ymin, xmax, ymax, w_r, h_r), fill=0, dtype="uint16")
    col = np.clip(((gx - xmin) / dx * (w_r - 1)).astype(int), 0, w_r - 1)
    row = np.clip(((ymax - gy) / dy * (h_r - 1)).astype(int), 0, h_r - 1)
    idx_geo = rast[row, col]
    pal = np.vstack([[110, 110, 120]] + [hex_rgb(c) for c in geo["cor"]])
    cor_geo = pal[idx_geo].reshape(-1, 3).astype(np.uint8)
    sem = idx_geo.ravel() == 0
    cor_geo[sem] = cor_hip[sem]

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(**base, vertexcolor=cor_hip, name="terreno", showlegend=False, hoverinfo="skip", meta="terreno",
                            lighting=dict(ambient=0.85, diffuse=0.45, specular=0.02)))

    # ---- modelo regional indicativo das unidades (tendencia de Taio extrapolada; calibra nos topos se houver CSV) ----
    tilt = v.Z_REF_TILT + v.TREND_A * (gx - v.TREND_X0) + v.TREND_B * (gy - v.TREND_Y0)
    prof = v.PROFUNDIDADE_CAMADAS  # 0, 350, 430, 485, 585, 854
    contatos = [gz if m == 0 else np.minimum(tilt - prof[m], gz) for m in range(len(prof))]
    calibrado = False
    if TOPOS_CSV.exists():
        tp = pd.read_csv(TOPOS_CSV)
        pt_xy = {r.nome: (r.geometry.x, r.geometry.y) for r in sel.itertuples()}

        def calibrar(k, formacao):
            # superficie de ordem k (topo da unidade k = base da anterior) ajustada aos topos medidos nos pocos
            sub = tp[(tp["formacao"] == formacao) & tp["prof_topo_m"].notna() & tp["poco"].isin(pt_xy)]
            if len(sub) < 3:
                return False
            px = np.array([pt_xy[p][0] for p in sub["poco"]]); py = np.array([pt_xy[p][1] for p in sub["poco"]])
            ztop = z_em(px, py) - sub["prof_topo_m"].values
            ttilt = v.Z_REF_TILT + v.TREND_A * (px - v.TREND_X0) + v.TREND_B * (py - v.TREND_Y0) - prof[k]
            resid = griddata((px, py), ztop - ttilt, (gx, gy), method="linear")
            resid = np.where(np.isnan(resid), griddata((px, py), ztop - ttilt, (gx, gy), method="nearest"), resid)
            contatos[k] = np.minimum(tilt - prof[k] + resid, gz)
            return True

        for k, nome in enumerate(v.NOMES_CAMADAS):
            calibrado |= calibrar(k, nome)
        # base do modelo = base do Rio Bonito (ficha da planilha); sem ela a base fica invertida sob os topos calibrados
        calibrado |= calibrar(len(prof) - 1, "Rio Bonito (base)")
        # ordem estratigrafica: nenhuma superficie pode ficar acima da anterior (evita camadas com espessura negativa)
        for k in range(1, len(contatos)):
            contatos[k] = np.minimum(contatos[k], contatos[k - 1])
    for m, nome in enumerate(v.NOMES_CAMADAS):
        z = np.where(contatos[m + 1] < gz - 8, contatos[m + 1], np.nan)
        fig.add_trace(go.Surface(x=gx, y=gy, z=z, colorscale=[[0, COR_UNIDADE[nome]], [1, COR_UNIDADE[nome]]], showscale=False, opacity=0.55,
                                 name=f"Base {nome}", showlegend=True, hoverinfo="name", meta="unidades", visible=False,
                                 lighting=dict(ambient=0.9, diffuse=0.3)))
    # paredes e chao do bloco (empilhamento ate a base do modelo regional)
    ri = np.unique(np.append(np.arange(0, ny, 3), ny - 1)); ci = np.unique(np.append(np.arange(0, nx, 3), nx - 1))
    sub = np.ix_(ri, ci)
    for k, nome in enumerate(v.NOMES_CAMADAS):
        px_, py_, pz_, pi_, pj_, pk_ = v.paredes_caixa(gx[sub], gy[sub], contatos[k][sub], contatos[k + 1][sub])
        fig.add_trace(go.Mesh3d(x=np.round(px_, 0), y=np.round(py_, 0), z=np.round(pz_, 0), i=pi_, j=pj_, k=pk_, color=COR_UNIDADE[nome],
                                flatshading=False, name=nome, legendgroup=nome, showlegend=False, hoverinfo="name", meta="paredes",
                                lighting=dict(ambient=0.95, diffuse=0.1)))
    fig.add_trace(go.Mesh3d(x=np.round(gx.ravel(), 0), y=np.round(gy.ravel(), 0), z=np.round(contatos[-1].ravel(), 0), i=ti, j=tj, k=tk,
                            color=v.CORES_CAMADAS[-1], flatshading=False, showlegend=False, hoverinfo="skip", meta="paredes",
                            lighting=dict(ambient=0.95, diffuse=0.1)))

    # ---- linhas: lineamentos locais, falhas/fraturas/diques CPRM ----
    janela = box(xmin, ymin, xmax, ymax)

    def linhas(geoms, passo, extra):
        ex, ey, ez = [], [], []
        for g in geoms:
            inter = g.intersection(janela)
            for seg in ([inter] if inter.geom_type == "LineString" else list(getattr(inter, "geoms", []))):
                if seg.geom_type != "LineString" or seg.length < 500:
                    continue
                pts = [seg.interpolate(d) for d in np.linspace(0, seg.length, max(int(seg.length // passo), 2))]
                px, py = [q.x for q in pts], [q.y for q in pts]
                ex += px + [None]; ey += py + [None]; ez += list(z_em(px, py, extra)) + [None]
        return ex, ey, ez

    est = gpd.read_file(GEOLOGIA_REGIONAL_GPKG, layer="estrutura")
    dq = gpd.read_file(GEOLOGIA_REGIONAL_GPKG, layer="dique")
    lin = gpd.read_file(LINEAMENTOS_GPKG).to_crs(CRS_UTM)
    for nome, geoms, cor, larg, meta in (
            ("Falha (CPRM)", est.loc[est["categoria"] == "Falha", "geometry"], AZUL_FALHA, 5, "falha"),
            ("Fratura (CPRM)", est.loc[est["categoria"] == "Fratura", "geometry"], ROSA_FRATURA, 3, "fratura"),
            ("Diques (CPRM)", dq.geometry, v.COR_DIQUE, 6, "dique"),
            ("Lineamento positivo", lin.loc[lin["tipo"] == "Positivo", "geometry"], "#2E7D5B", 4, "lin"),
            ("Lineamento negativo", lin.loc[lin["tipo"] == "Negativo", "geometry"], "#E8A33D", 4, "lin")):
        ex, ey, ez = linhas(geoms, 1500, 60)
        if ex:
            fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(color=cor, width=larg), name=nome, meta=meta, hoverinfo="name"))

    # ---- area de estudo e raio dos pocos ----
    tx0, ty0, tx1, ty1 = TAIO_AREA
    bx, by = [tx0, tx1, tx1, tx0, tx0], [ty0, ty0, ty1, ty1, ty0]
    fig.add_trace(go.Scatter3d(x=bx, y=by, z=z_em(bx, by, 200), mode="lines", line=dict(color="#FFFFFF", width=5), name="Área de estudo (Taió)",
                               meta="area", hoverinfo="name"))
    ang = np.linspace(0, 2 * np.pi, 181)
    cxs, cys = CENTRO_TAIO[0] + raio_km * 1000 * np.cos(ang), CENTRO_TAIO[1] + raio_km * 1000 * np.sin(ang)
    dentro = (cxs >= xmin) & (cxs <= xmax) & (cys >= ymin) & (cys <= ymax)
    cxs, cys = np.where(dentro, cxs, np.nan), np.where(dentro, cys, np.nan)
    fig.add_trace(go.Scatter3d(x=cxs, y=cys, z=np.nan_to_num(z_em(np.nan_to_num(cxs), np.nan_to_num(cys), 150)), mode="lines",
                               line=dict(color="#13BBE6", width=4, dash="dash"), name=f"Raio dos 10 poços ({raio_km:.0f} km)", meta="area", hoverinfo="name"))

    # ---- pocos: haste da boca (relevo) ate o fundo (profundidade total) ----
    zbocas = z_em(sel.geometry.x.values, sel.geometry.y.values)
    fundos = zbocas - sel["profundidade_m"].fillna(0).values
    hx, hy, hz = [], [], []
    for x, y, z0, z1 in zip(sel.geometry.x, sel.geometry.y, zbocas, fundos):
        hx += [x, x, None]; hy += [y, y, None]; hz += [z0, z1, None]
    fig.add_trace(go.Scatter3d(x=hx, y=hy, z=hz, mode="lines", line=dict(color="#FFFFFF", width=6), name="Poço (boca → fundo)", meta="pocos", hoverinfo="skip"))
    texto = [f"<b>{r.nome}</b><br>{r.fonte}<br>Boca ≈ {z0:.0f} m · Profundidade total {r.profundidade_m:.0f} m<br>Fundo ≈ {z1:.0f} m (cota)"
             f"<br>{r.dist_km:.0f} km de Taió" for r, z0, z1 in zip(sel.itertuples(), zbocas, fundos)]
    fig.add_trace(go.Scatter3d(x=sel.geometry.x, y=sel.geometry.y, z=zbocas + 20, mode="markers+text", marker=dict(size=6, symbol="diamond", color="#FFFFFF", line=dict(width=1, color="#13BBE6")),
                               text=[n.split(" (")[0] for n in sel["nome"]], textposition="top center", textfont=dict(size=10, color="#FFFFFF"),
                               hovertext=texto, hoverinfo="text", name="Bocas dos poços", meta="pocos"))
    fig.add_trace(go.Scatter3d(x=sel.geometry.x, y=sel.geometry.y, z=fundos, mode="markers", marker=dict(size=5, symbol="square", color="#FFD54A"),
                               hovertext=texto, hoverinfo="text", name="Fundo do poço (TD)", meta="pocos"))

    zmin_dado = min(float(np.nanmin(contatos[-1])), float(np.nanmin(fundos)))
    zmax = float(gz.max()) + 200
    zmin = zmin_dado - 150
    fig.update_layout(
        paper_bgcolor=v.MARCA_NAVY, font=dict(family=v.MARCA_FONTE, color=v.MARCA_CINZA_CLARO), margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(bgcolor="rgba(12,41,69,0.8)", bordercolor=v.MARCA_ROXO, borderwidth=1, x=0.01, y=0.99),
        scene=dict(
            xaxis=dict(title="X (UTM)", range=[xmin - 0.02 * dx, xmax + 0.02 * dx], backgroundcolor="#0C2945", gridcolor="#2A4663", color=v.MARCA_CINZA_CLARO),
            yaxis=dict(title="Y (UTM)", range=[ymin - 0.02 * dy, ymax + 0.02 * dy], backgroundcolor="#0C2945", gridcolor="#2A4663", color=v.MARCA_CINZA_CLARO),
            zaxis=dict(title="Cota (m)", range=[zmin, zmax], backgroundcolor="#0C2945", gridcolor="#2A4663", color=v.MARCA_CINZA_CLARO),
            aspectmode="manual", aspectratio=dict(x=1.04, y=1.04 * dy / dx, z=(zmax - zmin) * 1.0 / dx),
            camera=dict(eye=dict(x=-1.0, y=-1.5, z=0.8)),
        ),
    )
    ar = fig.layout.scene.aspectratio
    cores = dict(hip=base64.b64encode(cor_hip.tobytes()).decode(), geo=base64.b64encode(cor_geo.tobytes()).decode())
    tabela = [dict(nome=r.nome, fonte=r.fonte.split(" (")[0], dist=round(r.dist_km, 1), td=None if pd.isna(r.profundidade_m) else round(float(r.profundidade_m)),
                   boca=round(float(z0)), fundo=round(float(z1))) for r, z0, z1 in zip(sel.itertuples(), zbocas, fundos)]
    logo = base64.b64encode(LOGO_PATH.read_bytes()).decode() if LOGO_PATH.exists() else ""
    aviso = ("Superfícies das unidades CALIBRADAS nos topos informados." if calibrado else
             "Superfícies indicativas: extrapolação do modelo regional de Taió (sem calibração nos poços — a tabela ANP não traz topos/litologia).")
    html = f"""<!DOCTYPE html>
<html lang="pt-br"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recorte regional dos furos — Taió Plumbing System</title>
<link rel="icon" type="image/png" href="assets/favicon.png?v=2"><link rel="shortcut icon" href="assets/favicon.ico?v=2">
<style>
 :root {{ --bg:{v.MARCA_NAVY}; --painel:{v.MARCA_ROXO_ESCURO}; --txt:{v.MARCA_CINZA_CLARO}; --ac:{v.MARCA_ROXO}; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; background:var(--bg); color:var(--txt); font-family:{v.MARCA_FONTE}; overflow:hidden; }}
 #barra {{ display:flex; flex-wrap:wrap; gap:6px 14px; align-items:center; padding:8px 70px 8px 14px; background:var(--painel); border-bottom:2px solid var(--ac); }}
 #barra h1 {{ font-size:15px; margin:0 8px 0 0; }} #barra h1 b {{ color:var(--ac); }}
 .grupo {{ display:flex; gap:5px; align-items:center; font-size:12px; }}
 .grupo label {{ opacity:.7; text-transform:uppercase; font-size:10px; letter-spacing:.06em; }}
 button, select {{ background:var(--bg); color:var(--txt); border:1px solid var(--ac); border-radius:5px; padding:4px 9px; font:inherit; font-size:12px; cursor:pointer; }}
 button.ativo {{ background:var(--ac); color:var(--bg); }}
 #aviso {{ width:100%; font-size:11px; opacity:.75; }}
 #corpo {{ display:flex; }} #plot {{ flex:1; min-width:0; }}
 #lateral {{ width:330px; background:var(--painel); border-left:2px solid var(--ac); padding:10px 12px; overflow-y:auto; font-size:11px; }}
 #lateral h2 {{ font-size:12px; margin:0 0 6px 0; text-transform:uppercase; letter-spacing:.05em; color:var(--ac); }}
 table {{ border-collapse:collapse; width:100%; }} td, th {{ padding:3px 4px; border-bottom:1px solid #2A4663; text-align:right; }} td:first-child, th:first-child {{ text-align:left; }}
 #logo {{ position:fixed; top:8px; right:12px; width:46px; height:46px; border-radius:50%; border:2px solid var(--ac); z-index:5; }}
</style></head><body>
<img id="logo" src="data:image/png;base64,{logo}">
<div id="barra"><h1><b>Recorte regional</b> — furos de Taió</h1>
 <div class="grupo"><label>Cor</label><button class="tex ativo" data-t="hip">Hipsometria</button><button class="tex" data-t="geo">Geologia SC (CPRM)</button></div>
 <div class="grupo"><label>Camadas</label><button class="cam ativo" data-m="paredes">Bloco (unidades)</button><button class="cam" data-m="unidades">Superfícies das unidades</button><button class="cam ativo" data-m="pocos">Poços</button><button class="cam ativo" data-m="area">Área / raio</button></div>
 <div class="grupo"><label>Estruturas</label><button class="cam ativo" data-m="falha">Falha CPRM</button><button class="cam ativo" data-m="fratura">Fratura CPRM</button><button class="cam" data-m="dique">Diques CPRM</button><button class="cam" data-m="lin">Lineam. locais</button></div>
 <div class="grupo"><label>Exagero vertical</label><select id="exag"><option>1</option><option>5</option><option selected>10</option><option>20</option><option>40</option></select></div>
 <div id="aviso">{aviso}</div></div>
<div id="corpo"><div id="plot"></div>
<aside id="lateral"><h2>Poços no recorte</h2><table id="tab"><tr><th>Poço</th><th>km</th><th>TD (m)</th><th>Fundo (cota)</th></tr></table>
 <p style="opacity:.7;line-height:1.5">Cota do fundo = relevo (Terrain Tiles) na boca − profundidade total. Unidades calibradas nos topos da planilha PMP/ANP (Irati, Palermo, Rio Bonito e base do Rio Bonito) para os poços do recorte que a planilha cobre; Teresina e Serra Alta não têm topo separado na fonte e seguem a tendência regional. Longe dos poços as superfícies são extrapoladas.</p></aside></div>
<script>{get_plotlyjs()}</script>
<script>
var FIG = {fig.to_json()}, AR = {{x: {ar.x}, y: {ar.y}, z: {ar.z}}}, CORES = {json.dumps(cores)}, TAB = {json.dumps(tabela, ensure_ascii=False)};
var gd = document.getElementById('plot'), textura = 'hip', EXAG = {EXAGERO_PADRAO}, on = {{paredes: true, unidades: false, pocos: true, area: true, falha: true, fratura: true, dique: false, lin: false}};
function altura() {{ return window.innerHeight - document.getElementById('barra').offsetHeight; }}
function b64u8(s) {{ var b = atob(s), a = new Uint8Array(b.length); for (var i = 0; i < b.length; i++) a[i] = b.charCodeAt(i); return a; }}
var cache = {{}};
function cor(t) {{ if (!cache[t]) {{ var u = b64u8(CORES[t]), a = []; for (var i = 0; i < u.length; i += 3) a.push([u[i], u[i + 1], u[i + 2]]); cache[t] = a; }} return cache[t]; }}
var idxT = FIG.data.findIndex(function(t) {{ return t.meta === 'terreno'; }});
function aplicar() {{
  var vis = FIG.data.map(function(t) {{ var m = t.meta; return m && on.hasOwnProperty(m) ? on[m] : true; }});
  Plotly.restyle(gd, {{visible: vis}});
  Plotly.restyle(gd, {{vertexcolor: [cor(textura)]}}, [idxT]);
  Plotly.relayout(gd, {{'scene.aspectratio': {{x: AR.x, y: AR.y, z: AR.z * EXAG}}}});
}}
function dimensionar() {{ gd.style.height = altura() + 'px'; document.getElementById('lateral').style.height = altura() + 'px'; }}
dimensionar();
Plotly.newPlot(gd, FIG.data, FIG.layout, {{responsive: true, displaylogo: false}}).then(aplicar);
document.querySelectorAll('.tex').forEach(function(b) {{ b.onclick = function() {{ textura = b.dataset.t;
  document.querySelectorAll('.tex').forEach(function(x) {{ x.classList.toggle('ativo', x === b); }}); aplicar(); }}; }});
document.querySelectorAll('.cam').forEach(function(b) {{ b.onclick = function() {{ on[b.dataset.m] = !on[b.dataset.m]; b.classList.toggle('ativo', on[b.dataset.m]); aplicar(); }}; }});
document.getElementById('exag').onchange = function() {{ EXAG = parseFloat(this.value); aplicar(); }};
var tab = document.getElementById('tab');
TAB.forEach(function(r) {{ var tr = document.createElement('tr'); tr.innerHTML = '<td>' + r.nome.split(' (')[0] + '</td><td>' + r.dist + '</td><td>' + (r.td === null ? '—' : r.td) + '</td><td>' + r.fundo + '</td>'; tab.appendChild(tr); }});
window.addEventListener('resize', function() {{ dimensionar(); Plotly.Plots.resize(gd); }});
</script></body></html>"""
    OUT_HTML.write_text(html, encoding="utf-8")
    sys.path.insert(0, str(Path(__file__).parent)) if 'sys' in globals() else __import__('sys').path.insert(0, str(Path(__file__).parent))
    import tps_i18n  # botao de idioma (PT/EN) e volta ao hub
    tps_i18n.injetar(OUT_HTML, css="#barra{padding-right:200px}")
    print(f"Salvo em: {OUT_HTML} ({OUT_HTML.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
