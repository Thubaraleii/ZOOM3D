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
GEOLOGIA_REGIONAL_GPKG = BANCO / "dados_base" / "geologia_regional_sc.gpkg"
ESTRUTURAL_GPKG = BANCO / "Unificação" / "GPKG_Novos" / "nuvem_pontos_direcoes.gpkg"
LOGO_PATH = AQUI / "assets" / "logo_taio.png"
OUT_HTML = AQUI / "zooms_3d_taio.html"

# retangulos (xmin, ymin, xmax, ymax) em UTM 22S -- AJUSTAR conforme os recortes desejados
ZOOMS = {
    "A": dict(nome="Bela Vista", bbox=(589600, 7006200, 595800, 7015200)),
    "B": dict(nome="Pechincha", bbox=(586000, 7008900, 594200, 7014900)),
    "C": dict(nome="Domínio norte", bbox=(584000, 7015200, 588300, 7018400)),
    "D": dict(nome="Domínio sul", bbox=(582500, 6999700, 590100, 7009300)),
}
N_GRADE = 200          # vertices por eixo maior do relevo
OFFSET_DEC = {"Soleira": 10.0, "Dique": 12.0, "Depósito quaternário": 8.0}  # decalques em camadas (corpos por cima)
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
    return r[row, col], r


def mesh_xyz(m, **kw):
    vert, faces = m
    return dict(x=np.round(vert[:, 0], 1), y=np.round(vert[:, 1], 1), z=np.round(vert[:, 2], 1),
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], **kw)


def montar_zoom(chave, cfg, xyz, elevacao, lito, sat, sat_bbox, lineas, pontos, pocos, est_cprm, est_campo):
    bbox = cfg["bbox"]
    x0, y0, x1, y1 = bbox
    gx, gy, gz = grade_relevo(xyz, bbox, N_GRADE)
    ny, nx = gx.shape
    ti, tj, tk = triangulos_grade(ny, nx)
    base_xy = dict(x=np.round(gx.ravel(), 1), y=np.round(gy.ravel(), 1), z=np.round(gz.ravel(), 1), i=ti, j=tj, k=tk)
    fig = go.Figure()

    # --- relevo em 3 texturas (so uma visivel por vez) ---
    form, r_fine = raster_formacao(lito, bbox, gx, gy)
    sr, sg, sb = v.amostrar_satelite_rgb(sat, gx, gy, *sat_bbox)
    cor_sat = np.column_stack([sr.ravel(), sg.ravel(), sb.ravel()])
    zn = (gz.ravel() - gz.min()) / max(np.ptp(gz), 1.0)
    paleta_h = np.array([hex_rgb(c) for c in v.CORES_HIPSOMETRICAS])
    pos = zn * (len(paleta_h) - 1)
    lo = np.floor(pos).astype(int).clip(0, len(paleta_h) - 2)
    fr = (pos - lo)[:, None]
    cor_hip = (paleta_h[lo] * (1 - fr) + paleta_h[lo + 1] * fr).astype(np.uint8)
    fig.add_trace(go.Mesh3d(**base_xy, vertexcolor=cor_hip, name="terreno", visible=True, showlegend=False, hoverinfo="skip",
                            lighting=dict(ambient=0.7, diffuse=0.6, specular=0.05), flatshading=False, meta="terreno"))
    # decalques geologicos: poligonos exatos (contatos finos e suaves) recortados no bbox e triangulados com os nos da malha,
    # drapeados no relevo; os corpos intrusivos ficam um pouco acima do sedimentar (sem z-fighting)
    nos_zoom = np.column_stack([gx.ravel(), gy.ravel()])
    janela_dec = box(*bbox)
    for nome_f in PRIORIDADE:
        vs, fs, off_v = [], [], 0
        for g in lito.loc[lito["formacao"] == nome_f, "geometry"]:
            inter = g.intersection(janela_dec)
            for parte in ([inter] if inter.geom_type == "Polygon" else list(getattr(inter, "geoms", []))):
                if parte.geom_type != "Polygon" or parte.area < 150:
                    continue
                pxy, tri = v.triangular_interior(parte, 20.0, None, nos_zoom)
                if len(tri) == 0:
                    continue
                zz = np.array([elevacao(x, y) for x, y in pxy]) + OFFSET_DEC.get(nome_f, 6.0)
                vs.append(np.column_stack([pxy, zz])); fs.append(tri + off_v); off_v += len(pxy)
        if vs:
            vv, ff = np.vstack(vs), np.vstack(fs)
            fig.add_trace(go.Mesh3d(x=np.round(vv[:, 0], 1), y=np.round(vv[:, 1], 1), z=np.round(vv[:, 2], 1), i=ff[:, 0], j=ff[:, 1], k=ff[:, 2],
                                    color=COR_FORMACAO[nome_f], flatshading=False, showlegend=False, hoverinfo="skip", name=nome_f, meta="geo",
                                    lighting=dict(ambient=0.9, diffuse=0.3, specular=0.0)))

    # --- empilhamento (paredes do bloco) ---
    tilt = v.Z_REF_TILT + v.TREND_A * (gx - v.TREND_X0) + v.TREND_B * (gy - v.TREND_Y0)
    contatos = [gz if m == 0 else np.minimum(tilt - v.PROFUNDIDADE_CAMADAS[m], gz) for m in range(len(v.PROFUNDIDADE_CAMADAS))]
    contatos_orig = [c.copy() for c in contatos]  # sem o "erase": usados no chao do bloco e nas faces de corte
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
    zmin_bloco = float(np.nanmin(contatos_orig[-1]))
    # chao do bloco (fecha o solido embaixo) e placeholders das faces de corte (preenchidas em JS)
    fig.add_trace(go.Mesh3d(x=np.round(gx.ravel(), 1), y=np.round(gy.ravel(), 1), z=np.round(contatos_orig[-1].ravel(), 1),
                            i=ti, j=tj, k=tk, color=v.CORES_CAMADAS[-1], flatshading=False, showlegend=False, hoverinfo="skip",
                            lighting=dict(ambient=0.95, diffuse=0.1, specular=0.0)))
    for k_cap, (n_cap, c_cap) in enumerate(zip(list(v.NOMES_CAMADAS) + ["Soleira", "Dique"], list(v.CORES_CAMADAS) + [v.COR_SILL, v.COR_DIQUE])):
        fig.add_trace(go.Mesh3d(x=[0, 0, 0], y=[0, 0, 0], z=[0, 0, 0], i=[0], j=[1], k=[2], color=c_cap, flatshading=False,
                                showlegend=False, hoverinfo="name", name=n_cap, visible=False, meta=f"cap_{k_cap}",
                                lighting=dict(ambient=0.95, diffuse=0.1, specular=0.0)))

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
                                       legendgroup="lin", meta="lin_pos" if tipo == "Positivo" else "lin_neg", hoverinfo="name"))
    dentro = pontos[pontos.geometry.within(janela)]
    if len(dentro):
        fig.add_trace(go.Scatter3d(
            x=dentro.X, y=dentro.Y, z=[elevacao(x, y) + ELEV_PONTO for x, y in zip(dentro.X, dentro.Y)], mode="markers",
            marker=dict(size=4, color=[v.CORES_LITOLOGIA_CAMPO.get(l, v.COR_LITOLOGIA_PADRAO) for l in dentro["litologia_padronizada"]],
                        line=dict(width=1, color="#071729")),
            text=dentro["ponto_id"].astype(str), hoverinfo="text", name="Pontos de campo", meta="campo"))
    # falhas e fraturas: CPRM (falha azul, fratura rosa, linhas drapeadas) + falhas de campo (x pretos)
    for categoria, cor, larg in (("Falha", "#2563EB", 7), ("Fratura", "#D81B73", 5)):
        ex, ey, ez = [], [], []
        for g in est_cprm.loc[est_cprm["categoria"] == categoria, "geometry"]:
            inter = g.intersection(janela)
            for seg in ([inter] if inter.geom_type == "LineString" else list(getattr(inter, "geoms", []))):
                if seg.geom_type != "LineString" or seg.length < 20:
                    continue
                for d_seg in np.linspace(0, seg.length, max(int(seg.length // 40), 2)):
                    q = seg.interpolate(d_seg)
                    ex.append(q.x); ey.append(q.y); ez.append(elevacao(q.x, q.y) + ELEV_LINEAMENTO)
                ex.append(None); ey.append(None); ez.append(None)
        if ex:
            fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(color=cor, width=larg), name=f"{categoria} (CPRM)",
                                       legendgroup="estr", meta=f"est_{categoria.lower()}", hoverinfo="name"))
    if est_campo is not None:
        ec = est_campo[est_campo.geometry.within(janela)]
        if len(ec):
            fig.add_trace(go.Scatter3d(
                x=ec.geometry.x, y=ec.geometry.y, z=[elevacao(x, y) + ELEV_PONTO for x, y in zip(ec.geometry.x, ec.geometry.y)],
                mode="markers", marker=dict(size=4, symbol="x", color="#000000", line=dict(color="#FFFFFF", width=1)),
                text=[f"<b>{r.ponto_id}</b><br>{r.classificacao}" for r in ec.itertuples()], hoverinfo="text",
                name="Falha de campo", legendgroup="estr", meta="est_campo"))
            # linha de superficie (rumo) de cada falha de campo: segmento de 500 m ao longo do azimute/strike
            lx, ly, lz = [], [], []
            for r in ec.itertuples():
                if r.azimute_ou_strike_deg != r.azimute_ou_strike_deg:
                    continue
                ux, uy = np.sin(np.radians(r.azimute_ou_strike_deg)), np.cos(np.radians(r.azimute_ou_strike_deg))
                for f in np.linspace(-250, 250, 12):
                    px, py = r.geometry.x + f * ux, r.geometry.y + f * uy
                    lx.append(px); ly.append(py); lz.append(elevacao(px, py) + ELEV_LINEAMENTO)
                lx.append(None); ly.append(None); lz.append(None)
            fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", line=dict(color="#000000", width=3),
                                       name="Falha de campo (rumo)", legendgroup="estr", meta="est_campo", hoverinfo="name"))
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
    import base64 as _b64
    cap = dict(
        x0=x0, y0=y0, x1=x1, y1=y1, nx=int(nx), ny=int(ny), terr=np.round(gz.ravel(), 1).tolist(),
        con=[np.round(c.ravel(), 1).tolist() for c in contatos_orig],
        rw=int(r_fine.shape[1]), rh=int(r_fine.shape[0]), raster=_b64.b64encode(r_fine.astype(np.uint8).tobytes()).decode(),
        idx_sill=PRIORIDADE.index("Soleira") + 1, idx_dique=PRIORIDADE.index("Dique") + 1,
        esp_sill=v.ESPESSURA_MINIMA_SILL, base_dique=BASE_DIQUE_Z,
    )
    cores = dict(hip=_b64.b64encode(cor_hip.astype(np.uint8).tobytes()).decode(), sat=_b64.b64encode(cor_sat.astype(np.uint8).tobytes()).decode())
    info = dict(nome=cfg["nome"], dx=dx, dy=dy, dz=zmax - zmin, area_km2=dx * dy / 1e6,
                n_pontos=int(len(dentro)), n_pocos=int(len(pd_)), cap=cap, cores=cores)
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
    est_cprm = gpd.read_file(GEOLOGIA_REGIONAL_GPKG, layer="estrutura") if GEOLOGIA_REGIONAL_GPKG.exists() else None
    est_campo = None
    if ESTRUTURAL_GPKG.exists():
        est_campo = gpd.read_file(ESTRUTURAL_GPKG).to_crs(31982)
        est_campo = est_campo[est_campo["classificacao"].isin(["falha_ou_dique_ambiguo_relevo_negativo", "fratura_falha"])]
    figs, infos = {}, {}
    for chave, cfg in ZOOMS.items():
        print(f"  zoom {chave}: {cfg['nome']}")
        figs[chave], infos[chave] = montar_zoom(chave, cfg, xyz, elevacao, lito, sat, sat_bbox, lineas, pontos, pocos, est_cprm, est_campo)
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
 <div class="grupo"><label>Camadas</label><button class="cam ativo" data-m="campo">Pontos de campo</button><button class="cam ativo" data-m="pocos">Poços</button></div>
 <div class="grupo"><label>Estruturas</label><button class="cam ativo" data-m="est_falha">Falha CPRM</button><button class="cam ativo" data-m="est_fratura">Fratura CPRM</button><button class="cam ativo" data-m="est_campo">Falha de campo</button><button class="cam ativo" data-m="lin_pos">Lineam. (+)</button><button class="cam ativo" data-m="lin_neg">Lineam. (−)</button></div>
 <div class="grupo"><label>Corte</label><select id="ceixo"><option value="">Sem corte</option><option value="x">Leste–Oeste (X)</option><option value="y">Norte–Sul (Y)</option></select>
  <input type="range" id="cpos" min="5" max="95" value="50" style="width:130px"><button id="cinv">Inverter lado</button></div>
 <div id="info"></div></div>
{"".join(f'<div class="plot{" ativo" if i == 0 else ""}" id="plot-{c}"></div>' for i, c in enumerate(ZOOMS))}
<script>{get_plotlyjs()}</script>
<script>
var FIGS = {{{",".join(f'"{c}": {figs[c]}' for c in ZOOMS)}}};
var INFOS = {json.dumps(infos, ensure_ascii=False)};
var criados = {{}}, atual = 'A', textura = 'tex_geologia', EXAG = 3, camadasOn = {{campo: true, pocos: true, est_falha: true, est_fratura: true, est_campo: true, lin_pos: true, lin_neg: true}};
function altura() {{ return window.innerHeight - document.getElementById('barra').offsetHeight; }}
var CORTE = {{eixo: '', frac: 0.5, inv: false}}, CAPD = {{}};
function b64u8(s) {{ var b = atob(s), a = new Uint8Array(b.length); for (var i = 0; i < b.length; i++) a[i] = b.charCodeAt(i); return a; }}
function prep(c) {{ if (!CAPD[c]) {{ var d = INFOS[c].cap; d.rast = b64u8(d.raster); CAPD[c] = d; }} return CAPD[c]; }}
function bil(d, arr, x, y) {{
  var fx = (x - d.x0) / (d.x1 - d.x0) * (d.nx - 1), fy = (y - d.y0) / (d.y1 - d.y0) * (d.ny - 1);
  fx = Math.min(Math.max(fx, 0), d.nx - 1.001); fy = Math.min(Math.max(fy, 0), d.ny - 1.001);
  var i = Math.floor(fx), j = Math.floor(fy), tx = fx - i, ty = fy - j;
  function g(a, b) {{ return arr[b * d.nx + a]; }}
  return (g(i, j) * (1 - tx) + g(i + 1, j) * tx) * (1 - ty) + (g(i, j + 1) * (1 - tx) + g(i + 1, j + 1) * tx) * ty;
}}
function formAt(d, x, y) {{
  var col = Math.min(Math.max(Math.round((x - d.x0) / (d.x1 - d.x0) * (d.rw - 1)), 0), d.rw - 1);
  var row = Math.min(Math.max(Math.round((d.y1 - y) / (d.y1 - d.y0) * (d.rh - 1)), 0), d.rh - 1);
  return d.rast[row * d.rw + col];
}}
// face do corte: empilhamento (com a soleira/dique "apagando" o sedimentar) ao longo do plano de corte
function montarCaps(c, eixo, pos) {{
  var d = prep(c), ns = 420, tops = [], bots = [], X = [], Y = [], u, k, i;
  for (u = 0; u < 7; u++) {{ tops.push([]); bots.push([]); }}
  for (i = 0; i < ns; i++) {{
    var t = i / (ns - 1), x = eixo === 'x' ? pos : d.x0 + t * (d.x1 - d.x0), y = eixo === 'x' ? d.y0 + t * (d.y1 - d.y0) : pos;
    X.push(x); Y.push(y);
    var z0 = bil(d, d.terr, x, y), cs = d.con.map(function(a) {{ return bil(d, a, x, y); }}), f = formAt(d, x, y);
    var base = Math.min(cs[1], z0 - d.esp_sill);
    for (k = 0; k < 5; k++) {{
      var tp = cs[k], bt = cs[k + 1];
      if (f === d.idx_dique) {{ tp = NaN; bt = NaN; }}
      else if (f === d.idx_sill) {{ tp = Math.min(tp, base); bt = Math.min(bt, base); }}
      tops[k].push(tp); bots[k].push(bt);
    }}
    tops[5].push(f === d.idx_sill ? z0 : NaN); bots[5].push(f === d.idx_sill ? base : NaN);
    tops[6].push(f === d.idx_dique ? z0 : NaN); bots[6].push(f === d.idx_dique ? d.base_dique : NaN);
  }}
  var out = [];
  for (u = 0; u < 7; u++) {{
    var ok = [], ii = [], jj = [], kk = [], z = [];
    for (i = 0; i < ns; i++) ok.push(isFinite(tops[u][i]) && isFinite(bots[u][i]) && tops[u][i] - bots[u][i] > 0.3);
    for (i = 0; i < ns; i++) z.push(ok[i] ? tops[u][i] : 0);
    for (i = 0; i < ns; i++) z.push(ok[i] ? bots[u][i] : 0);
    for (i = 0; i < ns - 1; i++) if (ok[i] && ok[i + 1]) {{ ii.push(i, i + 1); jj.push(i + 1, ns + i + 1); kk.push(ns + i, ns + i); }}
    if (!ii.length) {{ ii = [0]; jj = [0]; kk = [0]; }}
    out.push({{x: X.concat(X), y: Y.concat(Y), z: z, i: ii, j: jj, k: kk}});
  }}
  return out;
}}
function aplicarCorte(c) {{
  var gd = document.getElementById('plot-' + c), fig = FIGS[c], s = fig.layout.scene, d = prep(c);
  var caps = [];
  fig.data.forEach(function(t, n) {{ if (t.meta && t.meta.indexOf('cap_') === 0) caps[parseInt(t.meta.slice(4), 10)] = n; }});
  var patch = {{'scene.xaxis.range': s.xaxis.range, 'scene.yaxis.range': s.yaxis.range}};
  if (!CORTE.eixo) {{
    Plotly.restyle(gd, {{visible: false}}, caps);
    Plotly.relayout(gd, patch);
    return;
  }}
  var eixo = CORTE.eixo, a0 = eixo === 'x' ? d.x0 : d.y0, a1 = eixo === 'x' ? d.x1 : d.y1, pos = a0 + CORTE.frac * (a1 - a0);
  var base = (eixo === 'x' ? s.xaxis.range : s.yaxis.range);
  patch['scene.' + eixo + 'axis.range'] = CORTE.inv ? [pos - 0.5, base[1]] : [base[0], pos + 0.5];
  Plotly.relayout(gd, patch);
  var dados = montarCaps(c, eixo, pos);
  dados.forEach(function(m, u) {{
    Plotly.restyle(gd, {{x: [m.x], y: [m.y], z: [m.z], i: [m.i], j: [m.j], k: [m.k], visible: true}}, [caps[u]]);
  }});
}}
var CORES = {{}};
function coresTerreno(c, tipo) {{
  var key = c + tipo;
  if (!CORES[key]) {{ var u = b64u8(INFOS[c].cores[tipo]), a = []; for (var i = 0; i < u.length; i += 3) a.push([u[i], u[i + 1], u[i + 2]]); CORES[key] = a; }}
  return CORES[key];
}}
function aplicar(c) {{
  var gd = document.getElementById('plot-' + c), fig = FIGS[c];
  var vis = fig.data.map(function(t) {{
    var m = t.meta;
    if (m === 'geo') return textura === 'tex_geologia';
    if (m && m.indexOf('cap_') === 0) return false;
    if (m && camadasOn.hasOwnProperty(m)) return camadasOn[m];
    return true;
  }});
  Plotly.restyle(gd, {{visible: vis}});
  var it = fig.data.findIndex(function(t) {{ return t.meta === 'terreno'; }});
  Plotly.restyle(gd, {{vertexcolor: [coresTerreno(c, textura === 'tex_satelite' ? 'sat' : 'hip')]}}, [it]);
  var s = fig.layout.scene, ar = s.aspectratio;
  Plotly.relayout(gd, {{'scene.aspectratio': {{x: ar.x, y: ar.y, z: ar.z * EXAG / 3}}}});
  aplicarCorte(c);
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
document.getElementById('ceixo').onchange = function() {{ CORTE.eixo = this.value; CORTE.frac = parseFloat(document.getElementById('cpos').value) / 100; aplicar(atual); }};
document.getElementById('cpos').oninput = function() {{ CORTE.frac = parseFloat(this.value) / 100; if (CORTE.eixo) aplicarCorte(atual); }};
document.getElementById('cinv').onclick = function() {{ CORTE.inv = !CORTE.inv; this.classList.toggle('ativo', CORTE.inv); if (CORTE.eixo) aplicarCorte(atual); }};
document.getElementById('exag').onchange = function() {{ EXAG = parseFloat(this.value); aplicar(atual); }};
window.addEventListener('resize', function() {{ var gd = document.getElementById('plot-' + atual); gd.style.height = altura() + 'px'; Plotly.Plots.resize(gd); }});
mostrar('A');
</script></body></html>"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Salvo em: {OUT_HTML} ({OUT_HTML.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
