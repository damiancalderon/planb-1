import pandas as pd

POLICE_PALETTE = [
    "#330000","#660000","#800000","#B22222","#E60000",
    "#FF1A1A","#FF4D4D","#FF8080","#FFB3B3","#FFE5E5"
]

MONTH_NAMES = {
    1:'Enero',2:'Febrero',3:'Marzo',4:'Abril',5:'Mayo',6:'Junio',
    7:'Julio',8:'Agosto',9:'Septiembre',10:'Octubre',11:'Noviembre',12:'Diciembre'
}

def _strip_accents_upper(text):
    if pd.isna(text): return text
    t = str(text)
    for a,b in (("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u")):
        t = t.replace(a,b).replace(a.upper(),b.upper())
    return t.upper()

def classify_crime(delito):
    d = str(delito).lower()
    if any(k in d for k in ['robo','fraude','confianza','extorsion','propiedad','despojo','asalto']): return 'Patrimony'
    if any(k in d for k in ['violacion','sexual','incesto','acoso']): return 'Freedom and Sexual Segurity'
    if any(k in d for k in ['secuestro','trafico','rapto','libertad']): return 'Personal Freedom'
    if any(k in d for k in ['homicidio','feminicidio','lesiones','golpes','menores','vida']): return 'Life and Integrity'
    if any(k in d for k in ['familia','familiar','genero']): return 'Family'
    if any(k in d for k in ['corrupcion de menores','trata de personas']): return 'Society'
    return 'Others'

def classify_violence(crime_classification, delito):
    violent_cls = ['Life and Integrity','Family','Personal Freedom','Freedom and Sexual Segurity']
    violent_kw = ['VIOLENCIA','ARMA','GOLPES','LESIONES','HOMICIDIO','FEMINICIDIO','SECUESTRO']
    if crime_classification in violent_cls: return 'Violent'
    if crime_classification in ['Patrimony','Others','Society']:
        return 'Violent' if any(k in str(delito).upper() for k in violent_kw) else 'Non-Violent'
    return 'Others'

def annotate_bars(ax):
    for p in ax.patches:
        h = p.get_height()
        if h and h > 0:
            ax.annotate(f'{h:.0f}', (p.get_x()+p.get_width()/2., h),
                        ha='center', va='center', xytext=(0,5),
                        textcoords='offset points', fontsize=8)

# estilo sin y-axis (para uso en EDA)
def pretty_ax(ax, hide_y=True):
    ax.grid(False); ax.set_facecolor("white")
    for side in ["top","right","left","bottom"]:
        ax.spines[side].set_visible(False)
    if hide_y:
        ax.set_yticks([]); ax.tick_params(left=False)

def cycle_palette(pal, n):
    import itertools
    if not pal: pal = ["#1f77b4"]
    return list(itertools.islice(itertools.cycle(pal), n))
