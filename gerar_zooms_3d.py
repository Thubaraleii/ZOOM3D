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
import pandas as pd
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
    "E": dict(nome="Horst", bbox=(581840, 6995690, 605090, 7031150), horst=True),
}
HORST_SHP = BANCO / "dados_base" / "horst.shp"
N_GRADE_HORST = 320    # vertices por eixo maior (bloco de ~23 x 35 km)
LARGURA_GRABEN = 3000.0   # largura (m) dos grabens subordinados de cada lado do horst (esquematico)
T_MAX_HORST = 300.0       # rejeito maximo do slider (m) na quina sul (esquematico)
A_HORST = 0.08            # fracao do rejeito na borda norte (quase parada)
G_GRABEN = 0.35           # afundamento dos grabens em relacao ao rejeito do horst
EXTRA_BASE = 300.0        # laje "base nao modelada" abaixo do pacote (so para as falhas descerem)
COR_BASE = "#5B6B7A"
COR_FALHA_HORST = "#13BBE6"
COR_FALHA_GRABEN = "#8FD3F0"
COR_BAIXO = "#9FB3C8"
N_GRADE = 200          # vertices por eixo maior do relevo
OFFSET_DEC = {"Depósito quaternário": 2.5, "Soleira": 3.5, "Dique": 4.5}  # decalques em camadas (corpos por cima)
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


def camadas_vetoriais(fig, janela, elevacao, lineas, pontos, pocos, est_cprm, est_campo, geoq):
    """Camadas ligaveis (lineamentos, campo, falhas/fraturas, geoquimica, pocos) drapeadas no relevo dentro da janela."""
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
            # linha de superficie (rumo) de cada falha de campo: segmento de 500 m ao longo do azimute/strike
            lx, ly, lz, lt = [], [], [], []
            for r in ec.itertuples():
                if r.azimute_ou_strike_deg != r.azimute_ou_strike_deg:
                    continue
                ux, uy = np.sin(np.radians(r.azimute_ou_strike_deg)), np.cos(np.radians(r.azimute_ou_strike_deg))
                for f in np.linspace(-250, 250, 12):
                    px, py = r.geometry.x + f * ux, r.geometry.y + f * uy
                    lx.append(px); ly.append(py); lz.append(elevacao(px, py) + ELEV_LINEAMENTO)
                    lt.append(f"<b>{r.ponto_id}</b><br>{r.classificacao}<br>Azimute: {r.azimute_ou_strike_deg:.0f}°")
                lx.append(None); ly.append(None); lz.append(None); lt.append(None)
            fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", line=dict(color="#000000", width=5),
                                       text=lt, name="Falha de campo", legendgroup="estr", meta="est_campo", hoverinfo="text"))
    gq = geoq[geoq.geometry.within(janela)]
    if len(gq):
        fig.add_trace(go.Scatter3d(
            x=gq.geometry.x, y=gq.geometry.y, z=[elevacao(x, y) + ELEV_PONTO + 8 for x, y in zip(gq.geometry.x, gq.geometry.y)], mode="markers",
            marker=dict(size=6, symbol="diamond", color=[v.COR_TI_ALTO if c == "Alto-Ti" else v.COR_TI_BAIXO for c in gq["classificacao_ti"]],
                        line=dict(width=1, color="#F1F5F9")),
            text=[f"<b>{r.amostra}</b> ({r.classificacao_ti})<br>{r.origem}<br>TiO₂ {r.TiO2:.2f}% · SiO₂ {r.SiO2:.1f}%" for r in gq.itertuples()],
            hoverinfo="text", name="Geoquímica (Alto/Baixo-Ti)", meta="geoq"))
    pd_ = pocos[pocos.geometry.within(janela)]
    if len(pd_):
        fig.add_trace(go.Scatter3d(
            x=pd_.geometry.x, y=pd_.geometry.y, z=[elevacao(p.x, p.y) + 5 for p in pd_.geometry], mode="markers+text",
            marker=dict(size=6, symbol="diamond", color="#FFFFFF"), text=pd_["nome"], hoverinfo="text", name="Poços", meta="pocos"))
    return dentro, gq, pd_


def montar_zoom(chave, cfg, xyz, elevacao, lito, sat, sat_bbox, lineas, pontos, pocos, est_cprm, est_campo, geoq):
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
                zz = np.array([elevacao(x, y) for x, y in pxy]) + OFFSET_DEC.get(nome_f, 1.5)
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

    dentro, gq, pd_ = camadas_vetoriais(fig, janela, elevacao, lineas, pontos, pocos, est_cprm, est_campo, geoq)

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
                n_pontos=int(len(dentro)), n_pocos=int(len(pd_)), n_geoq=int(len(gq)), cap=cap, cores=cores)
    return fig.to_json(), info


def grade_fn(gx, gy, g):
    """Interpolador bilinear (escalar ou vetorial) de uma grade regular (gx, gy, g)."""
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((gy[:, 0], gx[0, :]), g, bounds_error=False, fill_value=None)

    def fn(x, y):
        xa, ya = np.asarray(x, float), np.asarray(y, float)
        r = f(np.stack([np.atleast_1d(ya), np.atleast_1d(xa)], axis=-1))
        return float(r[0]) if xa.ndim == 0 else r
    return fn


def zonas_horst(horst_gdf):
    """Zonas do modelo esquematico a partir do shapefile do horst.
    Linha 'Alto' = 3 falhas que limitam o horst (vertices leste, sul, oeste, norte); linhas 'baixo' = extremidades.
    zona 1 = horst (bascula: ~nada na borda norte, maximo na quina sul); zona 2 = grabens subordinados nos dois flancos;
    zona 0 = restante do bloco (fixo)."""
    from shapely.geometry import LineString, Polygon
    alto = horst_gdf[horst_gdf["tipo"].str.lower() == "alto"].geometry.iloc[0]
    P = np.array([c[:2] for c in alto.coords])           # P1 (leste), P2 (sul), P3 (oeste), P4 (norte)
    H = Polygon(P).buffer(0)
    e = (P[0] - P[3]) / np.hypot(*(P[0] - P[3]))        # direcao da borda norte (P4 -> P1)
    n = np.array([-e[1], e[0]])
    if np.dot(P[1] - P[3], n) < 0:
        n = -n
    L = float(np.dot(P[1] - P[3], n))                    # distancia norte -> quina sul, perpendicular a borda norte
    band = Polygon([P[3] - 8e4 * e, P[3] + 8e4 * e, P[3] + 8e4 * e + L * n, P[3] - 8e4 * e + L * n])
    G = H.buffer(LARGURA_GRABEN, resolution=16).difference(H).intersection(band)
    from shapely.ops import unary_union
    G = unary_union(partes_poligono(G))
    baixo = [g for g in horst_gdf[horst_gdf["tipo"].str.lower() == "baixo"].geometry]
    return dict(H=H, G=G, band=band, P=P, e=e, n=n, L=L, baixo=baixo, alto=alto)


def classificar(Z, x, y):
    import shapely
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    xx, yy = np.where(ok, x, 0.0), np.where(ok, y, 0.0)
    z = np.zeros(len(xx), np.uint8)
    z[shapely.contains_xy(Z["G"], xx, yy)] = 2
    z[shapely.contains_xy(Z["H"], xx, yy)] = 1
    z[~ok] = 0
    return z


def partes_poligono(geom):
    return [g for g in ([geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))) if g.geom_type == "Polygon" and not g.is_empty]


def paredes_zonadas(bbox, Z, fn_topo, fn_base, passo=120.0):
    """Paredes do perimetro com amostras duplicadas nos cruzamentos com as falhas (degrau vertical, sem rampa)."""
    from shapely.geometry import LineString
    x0, y0, x1, y1 = bbox
    arestas = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    X, Y, Zz, I, J, K, ZN = [], [], [], [], [], [], []
    off = 0
    for a, b in arestas:
        linha = LineString([a, b]); Ln = linha.length
        ts = list(np.linspace(0, 1, int(Ln / passo) + 1))
        for poly in (Z["H"], Z["G"]):
            inter = linha.intersection(poly.boundary)
            for p in getattr(inter, "geoms", [inter]):
                if p.geom_type == "Point":
                    tc = linha.project(p) / Ln; d = 0.25 / Ln
                    ts += [tc - d, tc + d]
        ts = np.clip(np.sort(np.array(ts)), 0, 1)
        xs, ys = a[0] + ts * (b[0] - a[0]), a[1] + ts * (b[1] - a[1])
        n = len(ts)
        X.append(np.concatenate([xs, xs])); Y.append(np.concatenate([ys, ys]))
        Zz.append(np.concatenate([fn_topo(xs, ys), fn_base(xs, ys)]))
        zn = classificar(Z, xs, ys); ZN.append(np.concatenate([zn, zn]))
        idx = np.arange(n - 1)
        I.append(np.concatenate([off + idx, off + idx + 1])); J.append(np.concatenate([off + idx + 1, off + idx + n + 1])); K.append(np.concatenate([off + idx + n, off + idx + n]))
        off += 2 * n
    return (np.concatenate(X), np.concatenate(Y), np.concatenate(Zz), np.concatenate(I), np.concatenate(J), np.concatenate(K), np.concatenate(ZN))


def fita_vertical(linha, fn_topo, fn_base, passo=100.0, extra=6.0):
    """Plano de falha vertical (fita) ao longo de uma linha, do topo drapeado ate a base."""
    n = max(int(linha.length // passo), 2)
    pts = [linha.interpolate(d) for d in np.linspace(0, linha.length, n + 1)]
    xs, ys = np.array([q.x for q in pts]), np.array([q.y for q in pts])
    X, Y, Zz = np.concatenate([xs, xs]), np.concatenate([ys, ys]), np.concatenate([fn_topo(xs, ys) + extra, fn_base(xs, ys)])
    idx = np.arange(n)
    n1 = n + 1
    return X, Y, Zz, np.concatenate([idx, idx + 1]), np.concatenate([idx + 1, idx + n1 + 1]), np.concatenate([idx + n1, idx + n1])


def montar_horst(chave, cfg, xyz, elevacao, lito, sat, sat_bbox, lineas, pontos, pocos, est_cprm, est_campo, geoq, horst_gdf):
    """Aba E: horst de Taio. Bloco esquematico em 3 zonas rigidas (horst, grabens subordinados, restante) que o navegador
    desloca na vertical (horst basculando em torno da borda norte) -- malhas, camadas e face de corte seguem o mesmo campo."""
    import base64 as _b64
    from shapely.geometry import LineString
    bbox = cfg["bbox"]
    x0, y0, x1, y1 = bbox
    janela = box(*bbox)
    gx, gy, gz = grade_relevo(xyz, bbox, N_GRADE_HORST)
    ny, nx = gx.shape
    zb = grade_fn(gx, gy, gz)
    Z = zonas_horst(horst_gdf)
    zonas_poly = {1: Z["H"], 2: Z["G"], 0: janela.difference(Z["H"].union(Z["G"]))}
    fig = go.Figure()
    zon = {}

    def add(tr, z=None):
        fig.add_trace(tr)
        if z is not None:
            zon[len(fig.data) - 1] = np.asarray(z, np.uint8)

    nos = np.column_stack([gx.ravel(), gy.ravel()])
    # ---------------- relevo em pecas rigidas (uma por zona): bordas exatas nas falhas, sem rampa ----------------
    vs, fs, zs, off = [], [], [], 0
    for k, zp in zonas_poly.items():
        for parte in partes_poligono(zp.intersection(janela)):
            if parte.area < 2e3:
                continue
            pxy, tri = v.triangular_interior(parte, 40.0, None, nos)
            if len(tri) == 0:
                continue
            vs.append(pxy); fs.append(tri + off); zs.append(np.full(len(pxy), k, np.uint8)); off += len(pxy)
    PXY, TRI, ZN = np.vstack(vs), np.vstack(fs).astype(np.int32), np.concatenate(zs)
    zt = zb(PXY[:, 0], PXY[:, 1])
    form, r_fine = raster_formacao(lito, bbox, gx, gy)
    sr, sg, sb = v.amostrar_satelite_rgb(sat, PXY[:, 0], PXY[:, 1], *sat_bbox)
    cor_sat = np.column_stack([sr, sg, sb])
    zn_ = (zt - gz.min()) / max(np.ptp(gz), 1.0)
    paleta_h = np.array([hex_rgb(c) for c in v.CORES_HIPSOMETRICAS])
    pos = zn_ * (len(paleta_h) - 1)
    lo = np.floor(pos).astype(int).clip(0, len(paleta_h) - 2)
    fr = (pos - lo)[:, None]
    cor_hip = (paleta_h[lo] * (1 - fr) + paleta_h[lo + 1] * fr).astype(np.uint8)
    add(go.Mesh3d(x=np.round(PXY[:, 0], 1), y=np.round(PXY[:, 1], 1), z=np.round(zt, 1), i=TRI[:, 0], j=TRI[:, 1], k=TRI[:, 2],
                  vertexcolor=cor_hip, name="terreno", showlegend=False, hoverinfo="skip",
                  lighting=dict(ambient=0.7, diffuse=0.6, specular=0.05), flatshading=False, meta="terreno"), ZN)

    # ---------------- decalques geologicos por zona (sem triangulos cruzando as falhas) ----------------
    for nome_f in PRIORIDADE:
        vv_, ff_, zz_, off_v = [], [], [], 0
        for g in lito.loc[lito["formacao"] == nome_f, "geometry"]:
            for k, zp in zonas_poly.items():
                inter = g.intersection(zp.intersection(janela))
                for parte in partes_poligono(inter):
                    if parte.area < 150:
                        continue
                    pxy, tri = v.triangular_interior(parte, 20.0, None, nos)
                    if len(tri) == 0:
                        continue
                    zv = zb(pxy[:, 0], pxy[:, 1]) + OFFSET_DEC.get(nome_f, 1.5)
                    vv_.append(np.column_stack([pxy, zv])); ff_.append(tri + off_v); zz_.append(np.full(len(pxy), k, np.uint8)); off_v += len(pxy)
        if vv_:
            vv, ff = np.vstack(vv_), np.vstack(ff_).astype(np.int32)
            add(go.Mesh3d(x=np.round(vv[:, 0], 1), y=np.round(vv[:, 1], 1), z=np.round(vv[:, 2], 1), i=ff[:, 0], j=ff[:, 1], k=ff[:, 2],
                          color=COR_FORMACAO[nome_f], flatshading=False, showlegend=False, hoverinfo="skip", name=nome_f, meta="geo",
                          lighting=dict(ambient=0.9, diffuse=0.3, specular=0.0)), np.concatenate(zz_))

    # ---------------- empilhamento: paredes zonadas ----------------
    tilt = v.Z_REF_TILT + v.TREND_A * (gx - v.TREND_X0) + v.TREND_B * (gy - v.TREND_Y0)
    contatos = [gz if m == 0 else np.minimum(tilt - v.PROFUNDIDADE_CAMADAS[m], gz) for m in range(len(v.PROFUNDIDADE_CAMADAS))]
    contatos_orig = [c.copy() for c in contatos]
    m_sill = (form == PRIORIDADE.index("Soleira") + 1).reshape(gz.shape)
    m_dique = (form == PRIORIDADE.index("Dique") + 1).reshape(gz.shape)
    base_sill_g = np.minimum(contatos[1], gz - v.ESPESSURA_MINIMA_SILL)
    contatos = [np.where(m_dique, gz, np.where(m_sill, np.minimum(c, base_sill_g), c)) for c in contatos]
    fcon = [grade_fn(gx, gy, c) for c in contatos]
    fcon_orig = [grade_fn(gx, gy, c) for c in contatos_orig]
    for k, nome in enumerate(v.NOMES_CAMADAS):
        px, py, pz, pi, pj, pk, pzn = paredes_zonadas(bbox, Z, fcon[k], fcon[k + 1])
        add(go.Mesh3d(x=np.round(px, 1), y=np.round(py, 1), z=np.round(pz, 1), i=pi.astype(np.int32), j=pj.astype(np.int32), k=pk.astype(np.int32),
                      color=v.CORES_CAMADAS[k], flatshading=False, name=nome, legendgroup=nome, showlegend=True,
                      hoverinfo="name", lighting=dict(ambient=0.95, diffuse=0.1, specular=0.0, roughness=1.0)), pzn)
    # laje "base nao modelada": da base do pacote ate o chao -- so para as falhas descerem alem do modelo estratigrafico
    px, py, pz, pi, pj, pk, pzn = paredes_zonadas(bbox, Z, fcon_orig[5], lambda x, y: fcon_orig[5](x, y) - EXTRA_BASE)
    add(go.Mesh3d(x=np.round(px, 1), y=np.round(py, 1), z=np.round(pz, 1), i=pi.astype(np.int32), j=pj.astype(np.int32), k=pk.astype(np.int32),
                  color=COR_BASE, flatshading=False, name="Base (não modelada)", legendgroup="base", showlegend=True,
                  hoverinfo="name", lighting=dict(ambient=0.95, diffuse=0.1, specular=0.0, roughness=1.0)), pzn)
    zfloor = fcon_orig[5](PXY[:, 0], PXY[:, 1]) - EXTRA_BASE
    add(go.Mesh3d(x=np.round(PXY[:, 0], 1), y=np.round(PXY[:, 1], 1), z=np.round(zfloor, 1), i=TRI[:, 0], j=TRI[:, 1], k=TRI[:, 2],
                  color=COR_BASE, flatshading=False, showlegend=False, hoverinfo="skip",
                  lighting=dict(ambient=0.95, diffuse=0.1, specular=0.0)), ZN)
    nomes_cap = list(v.NOMES_CAMADAS) + ["Soleira", "Dique", "Base (não modelada)"]
    cores_cap = list(v.CORES_CAMADAS) + [v.COR_SILL, v.COR_DIQUE, COR_BASE]
    for k_cap, (n_cap, c_cap) in enumerate(zip(nomes_cap, cores_cap)):
        add(go.Mesh3d(x=[0, 0, 0], y=[0, 0, 0], z=[0, 0, 0], i=[0], j=[1], k=[2], color=c_cap, flatshading=False,
                      showlegend=False, hoverinfo="name", name=n_cap, visible=False, meta=f"cap_{k_cap}",
                      lighting=dict(ambient=0.95, diffuse=0.1, specular=0.0)))
    zmin_bloco = float(np.nanmin(contatos_orig[5])) - EXTRA_BASE

    # ---------------- paredes de falha: face exposta do bloco mais alto, coluna completa (camadas + base) ----------------
    def linhas_de(geom):
        return [g_ for g_ in ([geom] if geom.geom_type == "LineString" else list(getattr(geom, "geoms", []))) if g_.geom_type == "LineString" and g_.length > 50]
    interior = janela.buffer(-5)
    linhas_h = linhas_de(Z["H"].exterior.intersection(interior))                                              # lado do horst (sobe)
    linhas_r = linhas_de(Z["G"].boundary.difference(Z["H"].buffer(6)).intersection(interior))                 # lado fixo (grabens afundam)
    fbase = lambda x, y: fcon_orig[5](x, y) - EXTRA_BASE
    camadas_w = [(nome, cor, fcon[k], fcon[k + 1]) for k, (nome, cor) in enumerate(zip(v.NOMES_CAMADAS, v.CORES_CAMADAS))]
    camadas_w.append(("Base (não modelada)", COR_BASE, fcon_orig[5], fbase))
    for zcode, linhas in ((1, linhas_h), (0, linhas_r)):
        for nome, cor, ft, fb in camadas_w:
            XX, YY, ZZ, II, JJ, KK, off_w = [], [], [], [], [], [], 0
            for ln in linhas:
                rx, ry, rz, ri, rj, rk = fita_vertical(ln, ft, fb, 100.0, extra=0.0)
                XX.append(rx); YY.append(ry); ZZ.append(rz); II.append(ri + off_w); JJ.append(rj + off_w); KK.append(rk + off_w); off_w += len(rx)
            if not XX:
                continue
            XX, YY, ZZ = np.concatenate(XX), np.concatenate(YY), np.concatenate(ZZ)
            add(go.Mesh3d(x=np.round(XX, 1), y=np.round(YY, 1), z=np.round(ZZ, 1), i=np.concatenate(II).astype(np.int32), j=np.concatenate(JJ).astype(np.int32),
                          k=np.concatenate(KK).astype(np.int32), color=cor, flatshading=False, name=nome, legendgroup=nome, showlegend=False, hoverinfo="name",
                          lighting=dict(ambient=0.95, diffuse=0.1, specular=0.0, roughness=1.0)), np.full(len(XX), zcode, np.uint8))

    # ---------------- planos de falha (fitas verticais), deslocados 8 m para o lado de fora da parede (sem z-fighting) ----------------
    def deslocada(linha, alvo_dentro, d=8.0):
        for sinal in (d, -d):
            try:
                o = linha.offset_curve(sinal)
            except Exception:
                return linha
            if o.is_empty or o.geom_type != "LineString":
                continue
            m = o.interpolate(0.5, normalized=True)
            if alvo_dentro(m):
                return o
        return linha

    P = Z["P"]
    segs_alto = [LineString([P[0], P[1]]), LineString([P[1], P[2]]), LineString([P[2], P[3]])]
    segs_alto = [deslocada(s_, lambda m: not Z["H"].contains(m)) for s_ in segs_alto]
    for s_ in segs_alto:
        rx, ry, rz, ri, rj, rk = fita_vertical(s_, zb, fbase)
        add(go.Mesh3d(x=np.round(rx, 1), y=np.round(ry, 1), z=np.round(rz, 1), i=ri.astype(np.int32), j=rj.astype(np.int32), k=rk.astype(np.int32),
                      color=COR_FALHA_HORST, opacity=0.38, flatshading=True, showlegend=False, hoverinfo="name", name="Falha do horst (plano)",
                      meta="plano_falha", lighting=dict(ambient=1.0, diffuse=0.0)), np.zeros(len(rx), np.uint8))
    outer = Z["G"].boundary.difference(Z["H"].buffer(5)).difference(Z["band"].boundary.buffer(5)).intersection(janela)
    for s_ in [g for g in getattr(outer, "geoms", [outer]) if g.geom_type == "LineString" and g.length > 300]:
        s_ = deslocada(s_, lambda m: Z["G"].contains(m))
        rx, ry, rz, ri, rj, rk = fita_vertical(s_, zb, fbase, 150.0)
        add(go.Mesh3d(x=np.round(rx, 1), y=np.round(ry, 1), z=np.round(rz, 1), i=ri.astype(np.int32), j=rj.astype(np.int32), k=rk.astype(np.int32),
                      color=COR_FALHA_GRABEN, opacity=0.25, flatshading=True, showlegend=False, hoverinfo="name", name="Falha subordinada do graben (plano)",
                      meta="plano_falha", lighting=dict(ambient=1.0, diffuse=0.0)), np.zeros(len(rx), np.uint8))

    # ---------------- soleira e dique: pedacos por zona (rigidos) ----------------
    for nome, cor, kw in (("Soleira", v.COR_SILL, dict(base_fn=lambda x, y: v.elevacao_serra_alta(x, y, zb), espessura_minima=v.ESPESSURA_MINIMA_SILL)),
                          ("Dique", v.COR_DIQUE, dict(base_fn=lambda x, y: fcon_orig[5](x, y) - EXTRA_BASE + 2.0, espessura_minima=50.0))):
        mostrou = False
        for g in lito.loc[lito["formacao"] == nome, "geometry"]:
            for k, zp in zonas_poly.items():
                for p_ in partes_poligono(g.intersection(zp.intersection(janela))):
                    if p_.area < 400:
                        continue
                    vert, faces = v.construir_solido(p_, zb, **kw)
                    if len(faces) == 0:
                        continue
                    m_ = mesh_xyz((vert, faces)); m_["i"], m_["j"], m_["k"] = m_["i"].astype(np.int32), m_["j"].astype(np.int32), m_["k"].astype(np.int32)
                    add(go.Mesh3d(**m_, color=cor, flatshading=True, name=nome, legendgroup=nome, showlegend=not mostrou, hoverinfo="name",
                                  lighting=dict(ambient=0.7, diffuse=0.55)), np.full(len(vert), k, np.uint8))
                    mostrou = True

    # ---------------- camadas vetoriais (zona por ponto) ----------------
    i0 = len(fig.data)
    dentro, gq, pd_ = camadas_vetoriais(fig, janela, zb, lineas, pontos, pocos, est_cprm, est_campo, geoq)
    for idx in range(i0, len(fig.data)):
        tr = fig.data[idx]
        xs_ = np.array([np.nan if q is None else q for q in tr.x], float); ys_ = np.array([np.nan if q is None else q for q in tr.y], float)
        zon[idx] = classificar(Z, xs_, ys_)

    # ---------------- tracos das falhas do horst, extremidades ("baixo") e rotulos ----------------
    def linha_drapeada(geom, z_extra=14.0, passo=60.0):
        xs, ys = [], []
        for seg in ([geom] if geom.geom_type == "LineString" else list(getattr(geom, "geoms", []))):
            inter = seg.intersection(janela)
            for s2 in ([inter] if inter.geom_type == "LineString" else list(getattr(inter, "geoms", []))):
                if s2.geom_type != "LineString" or s2.length < 5:
                    continue
                pts = [s2.interpolate(d) for d in np.linspace(0, s2.length, max(int(s2.length // passo), 2))]
                xs += [q.x for q in pts] + [None]; ys += [q.y for q in pts] + [None]
        zs = [None if a is None else float(zb(a, b)) + z_extra for a, b in zip(xs, ys)]
        return xs, ys, zs

    xs, ys, zs = linha_drapeada(Z["alto"], 16.0)
    add(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=COR_FALHA_HORST, width=8), name="Falhas do horst", legendgroup="horst",
                     meta="linhas_horst", hoverinfo="name"), [1] * len(xs))
    for b_ in Z["baixo"]:
        xs, ys, zs = linha_drapeada(b_, 14.0)
        add(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=COR_BAIXO, width=6, dash="dash"), name="Limite do horst (baixo)", legendgroup="horst",
                         meta="linhas_horst", hoverinfo="name", showlegend=False), classificar(Z, [q if q is not None else np.nan for q in xs], [q if q is not None else np.nan for q in ys]))
    outer_l = Z["G"].boundary.difference(Z["H"].buffer(5)).difference(Z["band"].boundary.buffer(5))
    xs, ys, zs = linha_drapeada(outer_l, 14.0, 80.0)
    add(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=COR_FALHA_GRABEN, width=5, dash="dot"), name="Falhas subordinadas (grabens)", legendgroup="horst",
                     meta="linhas_horst", hoverinfo="name"), classificar(Z, [q if q is not None else np.nan for q in xs], [q if q is not None else np.nan for q in ys]))
    rot_x, rot_y, rot_t, rot_z = [], [], [], []
    ph = Z["H"].representative_point(); rot_x.append(ph.x); rot_y.append(ph.y); rot_t.append("HORST"); rot_z.append(1)
    for g in partes_poligono(Z["G"]):
        pg = g.representative_point(); rot_x.append(pg.x); rot_y.append(pg.y); rot_t.append("GRABEN"); rot_z.append(2)
    add(go.Scatter3d(x=rot_x, y=rot_y, z=[float(zb(a, b)) + 120 for a, b in zip(rot_x, rot_y)], mode="text", text=rot_t,
                     textfont=dict(size=16, color="#F1F5F9"), name="Rótulos", meta="rotulos_horst", hoverinfo="skip", showlegend=False), rot_z)

    # ---------------- layout, dados do corte e do deslocamento ----------------
    dx, dy = x1 - x0, y1 - y0
    zmax = float(np.nanmax(gz)) + 60 + T_MAX_HORST
    zmin = zmin_bloco - G_GRABEN * T_MAX_HORST - 10
    fig.update_layout(
        paper_bgcolor=v.MARCA_NAVY, font=dict(family=v.MARCA_FONTE, color=v.MARCA_CINZA_CLARO), margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(bgcolor="rgba(12,41,69,0.8)", bordercolor=v.MARCA_ROXO, borderwidth=1, x=0.01, y=0.99),
        scene=dict(
            xaxis=dict(title="X (UTM)", range=[x0 - 0.03 * dx, x1 + 0.03 * dx], backgroundcolor="#0C2945", gridcolor="#2A4663", color=v.MARCA_CINZA_CLARO),
            yaxis=dict(title="Y (UTM)", range=[y0 - 0.03 * dy, y1 + 0.03 * dy], backgroundcolor="#0C2945", gridcolor="#2A4663", color=v.MARCA_CINZA_CLARO),
            zaxis=dict(title="Z (m)", range=[zmin - 20, zmax], backgroundcolor="#0C2945", gridcolor="#2A4663", color=v.MARCA_CINZA_CLARO),
            aspectmode="manual", aspectratio=dict(x=1.0 * dx / max(dx, dy) * 1.06, y=1.06 * dy / max(dx, dy), z=(zmax - zmin + 20) * 3.0 / max(dx, dy) * 1.06),
            camera=dict(eye=dict(x=-1.35, y=-1.55, z=1.0)),
        ),
    )
    ringlist = lambda poly: [np.round(np.array(poly.exterior.coords), 1).tolist()] + [np.round(np.array(r.coords), 1).tolist() for r in poly.interiors]
    hz = dict(Tmax=T_MAX_HORST, px=float(Z["P"][3][0]), py=float(Z["P"][3][1]), nx=float(Z["n"][0]), ny=float(Z["n"][1]), L=Z["L"],
              a=A_HORST, g=G_GRABEN, tp=0.7, extra=EXTRA_BASE,
              H=ringlist(Z["H"]), G=[ringlist(g) for g in partes_poligono(Z["G"])],
              zonas={str(k_): _b64.b64encode(z_.tobytes()).decode() for k_, z_ in zon.items()})
    cap = dict(
        x0=x0, y0=y0, x1=x1, y1=y1, nx=int(nx), ny=int(ny), terr=np.round(gz.ravel(), 1).tolist(),
        con=[np.round(c.ravel(), 1).tolist() for c in contatos_orig],
        rw=int(r_fine.shape[1]), rh=int(r_fine.shape[0]), raster=_b64.b64encode(r_fine.astype(np.uint8).tobytes()).decode(),
        idx_sill=PRIORIDADE.index("Soleira") + 1, idx_dique=PRIORIDADE.index("Dique") + 1,
        esp_sill=v.ESPESSURA_MINIMA_SILL, base_dique=BASE_DIQUE_Z,
    )
    cores = dict(hip=_b64.b64encode(cor_hip.astype(np.uint8).tobytes()).decode(), sat=_b64.b64encode(cor_sat.astype(np.uint8).tobytes()).decode())
    info = dict(nome=cfg["nome"] + " (esquemático)", dx=dx, dy=dy, dz=zmax - zmin, area_km2=dx * dy / 1e6,
                n_pontos=int(len(dentro)), n_pocos=int(len(pd_)), n_geoq=int(len(gq)), cap=cap, cores=cores, hz=hz)
    return fig.to_json(), info


JS_HORST = r'''
// ---------------- horst (aba E): deslocamento rigido por zonas + face do corte com degraus nas falhas ----------------
var HZ = {frac: 0, anim: false}, ZONC = {};
function pipRing(r, x, y) {
  var c = false;
  for (var i = 0, j = r.length - 1; i < r.length; j = i++) {
    if (((r[i][1] > y) !== (r[j][1] > y)) && (x < (r[j][0] - r[i][0]) * (y - r[i][1]) / (r[j][1] - r[i][1]) + r[i][0])) c = !c;
  }
  return c;
}
function pipPoly(poly, x, y) { var n = 0; for (var q = 0; q < poly.length; q++) if (pipRing(poly[q], x, y)) n++; return n % 2 === 1; }
function zonaEm(h, x, y) { if (pipPoly(h.H, x, y)) return 1; for (var q = 0; q < h.G.length; q++) if (pipPoly(h.G[q], x, y)) return 2; return 0; }
// deslocamento vertical unitario (multiplicado pelo rejeito T): horst bascula (a na borda norte -> 1 na quina sul); grabens afundam
function dzUnit(h, zn, x, y) {
  if (!zn) return 0;
  var s = ((x - h.px) * h.nx + (y - h.py) * h.ny) / h.L; s = s < 0 ? 0 : (s > 1 ? 1 : s);
  var f = h.a + (1 - h.a) * s;
  if (zn === 1) return f;
  var t = (s - h.tp) / (1 - h.tp); t = t < 0 ? 0 : (t > 1 ? 1 : t); t = t * t * (3 - 2 * t);
  return -h.g * f * (1 - t);
}
function zonasDe(c) {
  if (!ZONC[c]) { var o = {}, z = INFOS[c].hz.zonas; Object.keys(z).forEach(function(k) { o[k] = b64u8(z[k]); }); ZONC[c] = o; }
  return ZONC[c];
}
function deslocarFigura(c) {
  var est = ESTZ[c], h = INFOS[c].hz, T = HZ.frac * h.Tmax, zc = zonasDe(c), idx = [], zs = [];
  Object.keys(est.orig).forEach(function(k) {
    var o = est.orig[k], zn = zc[k]; if (!zn) return;
    var n = o.x.length;
    if (!o.z0) {
      o.z0 = o.z; o.dzu = new Float32Array(n);
      for (var v = 0; v < n; v++) o.dzu[v] = (o.x[v] === null || o.z0[v] === null) ? 0 : dzUnit(h, zn[v], o.x[v], o.y[v]);
    }
    var z = Array.isArray(o.z0) ? new Array(n) : new Float64Array(n);
    for (var w = 0; w < n; w++) z[w] = o.z0[w] === null ? null : o.z0[w] + T * o.dzu[w];
    o.z = z; idx.push(parseInt(k, 10)); zs.push(z);
  });
  est.fracAplicada = HZ.frac;
  return {idx: idx, zs: zs};
}
// via slider/animacao: desloca e refaz o corte (se houver)
function aplicarHorst(c) {
  var est = ESTZ[c]; if (!est || !INFOS[c].hz) return;
  var r = deslocarFigura(c);
  if (CORTE.eixo || est.clipado) aplicarCorte(c);
  else Plotly.restyle(document.getElementById('plot-' + c), {z: r.zs}, r.idx);
}
var hzOcupado = false, hzPendente = false;
function agendarHorst() {
  if (hzOcupado) { hzPendente = true; return; }
  hzOcupado = true;
  setTimeout(function() { if (INFOS[atual].hz) aplicarHorst(atual); hzOcupado = false; if (hzPendente) { hzPendente = false; agendarHorst(); } }, 0);
}
function sincHorstUI() {
  var h = INFOS[atual].hz; if (!h) return;
  document.getElementById('hT').value = Math.round(HZ.frac * 100);
  document.getElementById('hTv').textContent = Math.round(HZ.frac * h.Tmax) + ' m';
}
// unica animacao do produto: horst sobe/desce
function animarHorst() {
  var btn = document.getElementById('hAnim');
  if (HZ.anim) { HZ.anim = false; btn.classList.remove('ativo'); return; }
  HZ.anim = true; btn.classList.add('ativo');
  var ini = HZ.frac, alvo = ini < 0.5 ? 1 : 0, p = 0, passos = 16;
  (function passo() {
    if (!HZ.anim || !INFOS[atual].hz) { HZ.anim = false; btn.classList.remove('ativo'); return; }
    p++; var u = p / passos, e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
    HZ.frac = ini + (alvo - ini) * e; sincHorstUI(); aplicarHorst(atual);
    if (p < passos) setTimeout(passo, 70); else { HZ.anim = false; btn.classList.remove('ativo'); }
  })();
}
function montarCapsH(c, eixo, pos) {
  var d = prep(c), h = d.hz, T = HZ.frac * h.Tmax, ns = 420, u, k, i;
  var ts = [], comp = eixo === 'x' ? (d.y1 - d.y0) : (d.x1 - d.x0), eps = 0.3 / comp;
  for (i = 0; i < ns; i++) ts.push(i / (ns - 1));
  [h.H].concat(h.G).forEach(function(poly) { poly.forEach(function(r) {
    for (var q = 0; q < r.length - 1; q++) {
      var a = r[q], b = r[q + 1], A = eixo === 'x' ? a[0] : a[1], B = eixo === 'x' ? b[0] : b[1];
      if ((A - pos) * (B - pos) < 0) {
        var uu = (pos - A) / (B - A), cc = eixo === 'x' ? a[1] + uu * (b[1] - a[1]) : a[0] + uu * (b[0] - a[0]);
        var tc = (cc - (eixo === 'x' ? d.y0 : d.x0)) / comp;
        if (tc > 0 && tc < 1) ts.push(tc - eps, tc + eps);
      }
    }
  }); });
  ts.sort(function(p, q) { return p - q; });
  var n = ts.length, tops = [], bots = [], X = [], Y = [];
  for (u = 0; u < 8; u++) { tops.push([]); bots.push([]); }
  for (i = 0; i < n; i++) {
    var t = Math.min(Math.max(ts[i], 0), 1);
    var x = eixo === 'x' ? pos : d.x0 + t * (d.x1 - d.x0), y = eixo === 'x' ? d.y0 + t * (d.y1 - d.y0) : pos;
    X.push(x); Y.push(y);
    var dz = T * dzUnit(h, zonaEm(h, x, y), x, y);
    var z0 = bil(d, d.terr, x, y), cs = d.con.map(function(a) { return bil(d, a, x, y); }), f = formAt(d, x, y);
    var base = Math.min(cs[1], z0 - d.esp_sill), chao = cs[5] - h.extra;
    for (k = 0; k < 5; k++) {
      var tp = cs[k], bt = cs[k + 1];
      if (f === d.idx_dique) { tp = NaN; bt = NaN; }
      else if (f === d.idx_sill) { tp = Math.min(tp, base); bt = Math.min(bt, base); }
      tops[k].push(tp + dz); bots[k].push(bt + dz);
    }
    tops[5].push(f === d.idx_sill ? z0 + dz : NaN); bots[5].push(f === d.idx_sill ? base + dz : NaN);
    tops[6].push(f === d.idx_dique ? z0 + dz : NaN); bots[6].push(f === d.idx_dique ? Math.max(d.base_dique, chao) + dz : NaN);
    tops[7].push(cs[5] + dz); bots[7].push(chao + dz);
  }
  var out = [];
  for (u = 0; u < 8; u++) {
    var ok = [], ii = [], jj = [], kk = [], z = [];
    for (i = 0; i < n; i++) ok.push(isFinite(tops[u][i]) && isFinite(bots[u][i]) && tops[u][i] - bots[u][i] > 0.3);
    for (i = 0; i < n; i++) z.push(ok[i] ? tops[u][i] : 0);
    for (i = 0; i < n; i++) z.push(ok[i] ? bots[u][i] : 0);
    for (i = 0; i < n - 1; i++) if (ok[i] && ok[i + 1]) { ii.push(i, i + 1); jj.push(i + 1, n + i + 1); kk.push(n + i, n + i); }
    if (!ii.length) { ii = [0]; jj = [0]; kk = [0]; }
    out.push({x: X.concat(X), y: Y.concat(Y), z: z, i: ii, j: jj, k: kk});
  }
  return out;
}
'''


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
    geoq = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=31982)
    if v.GEOQUIMICA_CSV.exists():
        dg = pd.read_csv(v.GEOQUIMICA_CSV)
        coord = {r.ponto_id: (r.geometry.x, r.geometry.y) for r in pontos.itertuples()}
        for i_, r in dg.iterrows():
            if pd.isna(r["utm_e"]) and pd.notna(r["ponto_id"]) and r["ponto_id"] in coord:
                dg.loc[i_, "utm_e"], dg.loc[i_, "utm_n"] = coord[r["ponto_id"]]
        dg = dg.dropna(subset=["utm_e", "utm_n"])
        geoq = gpd.GeoDataFrame(dg, geometry=gpd.points_from_xy(dg["utm_e"], dg["utm_n"]), crs=31982)
    horst_gdf = gpd.read_file(HORST_SHP).to_crs(31982)
    figs, infos = {}, {}
    for chave, cfg in ZOOMS.items():
        print(f"  zoom {chave}: {cfg['nome']}")
        if cfg.get("horst"):
            figs[chave], infos[chave] = montar_horst(chave, cfg, xyz, elevacao, lito, sat, sat_bbox, lineas, pontos, pocos, est_cprm, est_campo, geoq, horst_gdf)
        else:
            figs[chave], infos[chave] = montar_zoom(chave, cfg, xyz, elevacao, lito, sat, sat_bbox, lineas, pontos, pocos, est_cprm, est_campo, geoq)
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
 <div class="grupo"><label>Camadas</label><button class="cam ativo" data-m="campo">Pontos de campo</button><button class="cam ativo" data-m="geoq">Geoquímica (Alto/Baixo-Ti)</button><button class="cam ativo" data-m="pocos">Poços</button></div>
 <div class="grupo"><label>Estruturas</label><button class="cam ativo" data-m="est_falha">Falha CPRM</button><button class="cam ativo" data-m="est_fratura">Fratura CPRM</button><button class="cam ativo" data-m="est_campo">Falha de campo</button><button class="cam ativo" data-m="lin_pos">Lineam. (+)</button><button class="cam ativo" data-m="lin_neg">Lineam. (−)</button></div>
 <div class="grupo"><label>Corte</label><select id="ceixo"><option value="">Sem corte</option><option value="x">Leste–Oeste (X)</option><option value="y">Norte–Sul (Y)</option></select>
  <input type="range" id="cpos" min="5" max="95" value="50" style="width:130px"><button id="cinv">Inverter lado</button></div>
 <div class="grupo" id="g-horst" style="display:none"><label>Horst</label><input type="range" id="hT" min="0" max="100" value="0" style="width:150px"><span id="hTv" style="min-width:44px;text-align:right">0 m</span><button id="hAnim">Subir / descer</button><button class="cam ativo" data-m="plano_falha">Planos de falha</button><button class="cam ativo" data-m="linhas_horst">Traços</button><button class="cam ativo" data-m="rotulos_horst">Rótulos</button></div>
 <div class="grupo"><a href="regional_furos_3d.html"><button type="button">Recorte regional (furos) ↗</button></a></div>
 <div id="info"></div></div>
{"".join(f'<div class="plot{" ativo" if i == 0 else ""}" id="plot-{c}"></div>' for i, c in enumerate(ZOOMS))}
<script>{get_plotlyjs()}</script>
<script>
var FIGS = {{{",".join(f'"{c}": {figs[c]}' for c in ZOOMS)}}};
var INFOS = {json.dumps(infos, ensure_ascii=False)};
var criados = {{}}, atual = 'A', textura = 'tex_geologia', EXAG = 3, camadasOn = {{campo: true, geoq: true, pocos: true, est_falha: true, est_fratura: true, est_campo: true, lin_pos: true, lin_neg: true, plano_falha: true, linhas_horst: true, rotulos_horst: true}};
function altura() {{ return window.innerHeight - document.getElementById('barra').offsetHeight; }}
var CORTE = {{eixo: '', frac: 0.5, inv: false}}, CAPD = {{}};
function b64u8(s) {{ var b = atob(s), a = new Uint8Array(b.length); for (var i = 0; i < b.length; i++) a[i] = b.charCodeAt(i); return a; }}
function prep(c) {{ if (!CAPD[c]) {{ var d = INFOS[c].cap; d.rast = b64u8(d.raster); d.hz = INFOS[c].hz; CAPD[c] = d; }} return CAPD[c]; }}
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

{JS_HORST}
function dec(a) {{
  if (a && a.bdata) {{
    var T = {{i1: Int8Array, u1: Uint8Array, i2: Int16Array, u2: Uint16Array, i4: Int32Array, u4: Uint32Array, f4: Float32Array, f8: Float64Array}}[a.dtype];
    var b = atob(a.bdata), u = new Uint8Array(b.length); for (var q = 0; q < b.length; q++) u[q] = b.charCodeAt(q);
    return new T(u.buffer);
  }}
  return a;
}}
// recorte de malha por plano (x=pos ou y=pos): triangulos inteiros ficam, os cortados sao refeitos no plano -- a caixa NAO muda (corpo estatico)
function clipMesh(o, ax, pos, menor, vc) {{
  var P = ax === 'x' ? o.x : o.y, n = o.x.length, map = new Int32Array(n).fill(-1);
  var X = [], Y = [], Z = [], C = vc ? [] : null, I = [], J = [], K = [];
  function dentro(v) {{ return menor ? P[v] <= pos : P[v] >= pos; }}
  function add(v) {{ var m = map[v]; if (m < 0) {{ m = map[v] = X.length; X.push(o.x[v]); Y.push(o.y[v]); Z.push(o.z[v]); if (C) C.push(vc[v]); }} return m; }}
  function inter(a, b) {{
    var t = (pos - P[a]) / (P[b] - P[a]), m = X.length;
    X.push(o.x[a] + t * (o.x[b] - o.x[a])); Y.push(o.y[a] + t * (o.y[b] - o.y[a])); Z.push(o.z[a] + t * (o.z[b] - o.z[a]));
    if (C) C.push([vc[a][0] + t * (vc[b][0] - vc[a][0]), vc[a][1] + t * (vc[b][1] - vc[a][1]), vc[a][2] + t * (vc[b][2] - vc[a][2])]);
    return m;
  }}
  for (var q = 0, nt = o.i.length; q < nt; q++) {{
    var a = o.i[q], b = o.j[q], c = o.k[q], da = dentro(a), db = dentro(b), dc = dentro(c), s = da + db + dc;
    if (s === 3) {{ I.push(add(a)); J.push(add(b)); K.push(add(c)); }}
    else if (s > 0) {{
      var vs = [a, b, c], ds = [da, db, dc], poly = [];
      for (var e = 0; e < 3; e++) {{
        var p = vs[e], r = vs[(e + 1) % 3];
        if (ds[e]) poly.push(add(p));
        if (ds[e] !== ds[(e + 1) % 3]) poly.push(inter(p, r));
      }}
      for (var f = 1; f + 1 < poly.length; f++) {{ I.push(poly[0]); J.push(poly[f]); K.push(poly[f + 1]); }}
    }}
  }}
  if (!I.length) return {{x: [0, 0, 0], y: [0, 0, 0], z: [0, 0, 0], i: [0], j: [1], k: [2], vc: C ? [[0, 0, 0], [0, 0, 0], [0, 0, 0]] : null}};
  return {{x: X, y: Y, z: Z, i: I, j: J, k: K, vc: C}};
}}
function clipScat(o, ax, pos, menor) {{
  var P = ax === 'x' ? o.x : o.y, X = [], Y = [], Z = [];
  for (var q = 0, n = P.length; q < n; q++) {{
    var v = P[q], ok = v === null || (menor ? v <= pos : v >= pos);
    X.push(ok ? o.x[q] : null); Y.push(ok ? o.y[q] : null); Z.push(ok ? o.z[q] : null);
  }}
  return {{x: X, y: Y, z: Z}};
}}
// guarda os dados originais das malhas/linhas/pontos recortaveis
function guardarOrig(fig) {{
  var orig = {{}}, malhas = [], cenas = [];
  fig.data.forEach(function(t, n) {{
    if (t.type === 'mesh3d' && !(t.meta && t.meta.indexOf('cap_') === 0)) {{
      orig[n] = {{x: dec(t.x), y: dec(t.y), z: dec(t.z), i: dec(t.i), j: dec(t.j), k: dec(t.k)}}; malhas.push(n);
    }} else if (t.type === 'scatter3d' && t.meta && t.meta !== 'norte') {{
      orig[n] = {{x: dec(t.x), y: dec(t.y), z: dec(t.z)}}; cenas.push(n);
    }}
  }});
  return {{orig: orig, malhas: malhas, cenas: cenas, clipado: false}};
}}
// aplica (ou desfaz) o recorte numa figura; corpoCor = cores atuais do relevo (nested rgb)
function recortar(gd, est, idxTerreno, corpoCor, eixo, pos, menor) {{
  var rest = function(upd, idx) {{ if (idx.length) Plotly.restyle(gd, upd, idx); }};
  var corta = !!eixo;
  var outras = est.malhas.filter(function(n) {{ return n !== idxTerreno; }});
  var T = est.orig[idxTerreno];
  if (!corta) {{
    if (!est.clipado) return;
    Plotly.restyle(gd, {{x: [T.x], y: [T.y], z: [T.z], i: [T.i], j: [T.j], k: [T.k], vertexcolor: [corpoCor]}}, [idxTerreno]);
    var o = est.orig;
    rest({{x: outras.map(function(n) {{ return o[n].x; }}), y: outras.map(function(n) {{ return o[n].y; }}), z: outras.map(function(n) {{ return o[n].z; }}),
          i: outras.map(function(n) {{ return o[n].i; }}), j: outras.map(function(n) {{ return o[n].j; }}), k: outras.map(function(n) {{ return o[n].k; }})}}, outras);
    rest({{x: est.cenas.map(function(n) {{ return o[n].x; }}), y: est.cenas.map(function(n) {{ return o[n].y; }}), z: est.cenas.map(function(n) {{ return o[n].z; }})}}, est.cenas);
    est.clipado = false; return;
  }}
  var rt = clipMesh(T, eixo, pos, menor, corpoCor);
  Plotly.restyle(gd, {{x: [rt.x], y: [rt.y], z: [rt.z], i: [rt.i], j: [rt.j], k: [rt.k], vertexcolor: [rt.vc]}}, [idxTerreno]);
  var cm = outras.map(function(n) {{ return clipMesh(est.orig[n], eixo, pos, menor, null); }});
  rest({{x: cm.map(function(r) {{ return r.x; }}), y: cm.map(function(r) {{ return r.y; }}), z: cm.map(function(r) {{ return r.z; }}),
        i: cm.map(function(r) {{ return r.i; }}), j: cm.map(function(r) {{ return r.j; }}), k: cm.map(function(r) {{ return r.k; }})}}, outras);
  var cs = est.cenas.map(function(n) {{ return clipScat(est.orig[n], eixo, pos, menor); }});
  rest({{x: cs.map(function(r) {{ return r.x; }}), y: cs.map(function(r) {{ return r.y; }}), z: cs.map(function(r) {{ return r.z; }})}}, est.cenas);
  est.clipado = true;
}}
var ESTZ = {{}};
function aplicarCorte(c) {{
  var gd = document.getElementById('plot-' + c), fig = FIGS[c], d = prep(c);
  if (!ESTZ[c]) ESTZ[c] = guardarOrig(fig);
  var caps = []; fig.data.forEach(function(t, n) {{ if (t.meta && t.meta.indexOf('cap_') === 0) caps[parseInt(t.meta.slice(4), 10)] = n; }});
  var it = fig.data.findIndex(function(t) {{ return t.meta === 'terreno'; }});
  var corpoCor = coresTerreno(c, textura === 'tex_satelite' ? 'sat' : 'hip');
  if (!CORTE.eixo) {{ recortar(gd, ESTZ[c], it, corpoCor, '', 0, true); Plotly.restyle(gd, {{visible: false}}, caps); return; }}
  var eixo = CORTE.eixo, a0 = eixo === 'x' ? d.x0 : d.y0, a1 = eixo === 'x' ? d.x1 : d.y1, pos = a0 + CORTE.frac * (a1 - a0);
  recortar(gd, ESTZ[c], it, corpoCor, eixo, pos, !CORTE.inv);
  (d.hz ? montarCapsH : montarCaps)(c, eixo, pos).forEach(function(m, u) {{
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
  if (!(ESTZ[c] && ESTZ[c].clipado)) Plotly.restyle(gd, {{vertexcolor: [coresTerreno(c, textura === 'tex_satelite' ? 'sat' : 'hip')]}}, [it]);
  var s = fig.layout.scene, ar = s.aspectratio;
  Plotly.relayout(gd, {{'scene.aspectratio': {{x: ar.x, y: ar.y, z: ar.z * EXAG / 3}}}});
  if (INFOS[c].hz) {{
    if (!ESTZ[c]) ESTZ[c] = guardarOrig(fig);
    if (ESTZ[c].fracAplicada !== HZ.frac) {{ var r = deslocarFigura(c); if (!CORTE.eixo && !ESTZ[c].clipado) Plotly.restyle(gd, {{z: r.zs}}, r.idx); }}
  }}
  aplicarCorte(c);
}}
function mostrar(c) {{
  atual = c; HZ.anim = false; document.getElementById('hAnim').classList.remove('ativo');
  document.querySelectorAll('.plot').forEach(function(e) {{ e.classList.toggle('ativo', e.id === 'plot-' + c); }});
  document.querySelectorAll('.aba').forEach(function(b) {{ b.classList.toggle('ativo', b.dataset.z === c); }});
  var gd = document.getElementById('plot-' + c); gd.style.height = altura() + 'px';
  if (!criados[c]) {{
    Plotly.newPlot(gd, FIGS[c].data, FIGS[c].layout, {{responsive: true, displaylogo: false}}); criados[c] = true;
  }} else {{ Plotly.Plots.resize(gd); }}
  document.getElementById('g-horst').style.display = INFOS[c].hz ? 'flex' : 'none';
  sincHorstUI();
  aplicar(c);
  var i = INFOS[c];
  document.getElementById('info').textContent = i.nome + ' · ' + i.area_km2.toFixed(1) + ' km² · ' + i.n_pontos + ' pontos de campo · ' + i.n_geoq + ' amostras de geoquímica · ' + i.n_pocos + ' poços';
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
document.getElementById('hT').oninput = function() {{ HZ.frac = parseFloat(this.value) / 100; sincHorstUI(); agendarHorst(); }};
document.getElementById('hAnim').onclick = animarHorst;
window.addEventListener('resize', function() {{ var gd = document.getElementById('plot-' + atual); gd.style.height = altura() + 'px'; Plotly.Plots.resize(gd); }});
mostrar('A');
</script></body></html>"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Salvo em: {OUT_HTML} ({OUT_HTML.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
