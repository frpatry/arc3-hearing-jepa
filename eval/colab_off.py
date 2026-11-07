"""TESTS OFFICIELS SUR COLAB (les 25 jeux publics, comme sur Kaggle, sans dépenser le quota Kaggle).
Chaque réglage est coupé en 2 moitiés de jeux (processus parallèles sur la même carte) ; résultats dans <Drive>/arc3/runs/off/.
    python colab_off.py            lance tout en arrière-plan
    python colab_off.py --bilan    fusionne les moitiés : niveaux et score (règle officielle approchée) à 1000 / 2000 / 3000 actions
Réglages (26 oct., question user « lui donne-t-on assez de temps pour apprendre ? ») :
  A3k_win    modèle A complet, 3000 actions, INTÉRIORISER LA VICTOIRE (192 pas d'apprentissage juste après chaque niveau réussi)
  A3k_nowin  pareil sans intériorisation (pour isoler son effet)
  big3k_win  GRAND modèle complet (d384 L8), 3000 actions, intériorisation"""
import glob, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__)); D = "/content/drive/MyDrive/arc3"; R = f"{D}/runs/off"; ENV = f"{HERE}/data/environment_files"
CK = {"A": f"{D}/runs/A_actor.pt", "big": f"{D}/runs/big_full.pt"}
CFG = {"A3k_win": ("A", 3000, 192), "A3k_nowin": ("A", 3000, 0), "big3k_win": ("big", 3000, 192)}


def halves():
    g = sorted(os.listdir(ENV)); return [",".join(g[:13]), ",".join(g[13:])]


def launch():
    os.makedirs(R, exist_ok=True)
    for name, (m, B, w) in CFG.items():
        for i, gs in enumerate(halves()):
            cmd = (f"cd {HERE} && nohup python -u run_agent.py --env {ENV} --agent plan-grad --ckpt {CK[m]} --budget {B} --win_steps {w} "
                   f"--seed 0 --games {gs} --out {R}/{name}_{i}.json > {R}/{name}_{i}.log 2>&1 &")
            subprocess.run(cmd, shell=True); print("lancé", name, i, flush=True)


def score(g, B):                                     # règle officielle approchée (vérifiée à ±0.02 sur 2 tests Kaggle) : niveau i pèse i+1, (humain/agent)² plafonné à 1
    t = 0
    for x in g:
        if "level_at" not in x: continue
        la = [n for n in x["level_at"] if n <= B]; prev = 0; s = 0
        for i, n in enumerate(la): s += (i + 1) * min(1, (x["human"][i] / (n - prev)) ** 2); prev = n
        t += s / sum(i + 1 for i in range(x["win_levels"]))
    return t / 25 * 100


def bilan():
    for name in CFG:
        g = []; fini = 0
        for i in range(2):
            f = f"{R}/{name}_{i}.json"
            if os.path.exists(f): g += json.load(open(f))["games"]; fini += 1
            elif os.path.exists(f"{R}/{name}_{i}.log"):                   # moitié en cours : lignes déjà écrites
                g += [json.loads(l) for l in open(f"{R}/{name}_{i}.log") if l.startswith("{")]
        lv = {B: sum(len([n for n in x.get("level_at", []) if n <= B]) for x in g) for B in (1000, 2000, 3000)}
        err = [x["game"] for x in g if "error" in x]
        print(f"{name:10s} | {len(g)}/25 jeux ({fini}/2 moitiés finies) | niveaux @1000 {lv[1000]}  @2000 {lv[2000]}  @3000 {lv[3000]} | "
              f"score @1000 {score(g, 1000):.3f}  @3000 {score(g, 3000):.3f}" + (f" | ERREURS {err}" if err else ""), flush=True)
        print("   ", " ".join(f"{x['game']}:{x['level_at']}" for x in g if x.get("level_at")), flush=True)


if __name__ == "__main__":
    bilan() if "--bilan" in sys.argv else launch()
