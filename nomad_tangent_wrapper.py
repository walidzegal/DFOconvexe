"""
nomad_tangent_wrapper.py
=========================

Orchestration Python autour de l'executable C++ NOMAD (pollTangent.cpp),
sans passer par les bindings PyNomad (qui n'exposent pas le callback
USER_METHOD_POLL -- voir discussion). Ce module :

    1) compile pollTangent.cpp contre la bibliotheque NOMAD deja installee
       localement (variables d'environnement NOMAD_HOME / NOMAD_BUILD_DIR) ;
    2) lance l'executable resultant via subprocess ;
    3) parse la sortie DISPLAY_STATS de NOMAD ("bbe ( sol ) obj") en une
       structure Python (liste de dict / DataFrame pandas si disponible) ;
    4) expose une interface minimale calquee sur celle de PyNomad.optimize
       (x_best, f_best, nb_bb_evals, history) pour rester compatible avec
       tes scripts existants bases sur PyNomad.

Usage typique
-------------
    from nomad_tangent_wrapper import NomadTangentRunner

    runner = NomadTangentRunner(
        source_path="pollTangent.cpp",
        nomad_home="/chemin/vers/nomad",   # dossier racine du depot NOMAD compile
    )
    runner.compile()                        # une seule fois
    result = runner.run()

    print(result.f_best, result.x_best, result.nb_bb_evals)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

try:
    import pandas as pd  # optionnel : joli DataFrame pour l'historique
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


# --------------------------------------------------------------------------
# Structures de resultat
# --------------------------------------------------------------------------

@dataclass
class NomadStatLine:
    """Une ligne de la sortie DISPLAY_STATS de NOMAD : 'bbe ( x1 x2 ... ) f'."""
    bbe: int
    x: List[float]
    f: float


@dataclass
class NomadTangentResult:
    x_best: Optional[List[float]] = None
    f_best: Optional[float] = None
    nb_bb_evals: Optional[int] = None
    history: List[NomadStatLine] = field(default_factory=list)
    raw_stdout: str = ""
    raw_stderr: str = ""
    returncode: int = 0

    def history_dataframe(self):
        """Retourne l'historique sous forme de DataFrame pandas si possible,
        sinon une liste de dict."""
        rows = [{"bbe": h.bbe, "f": h.f, **{f"x{i}": xi for i, xi in enumerate(h.x)}}
                for h in self.history]
        if _HAS_PANDAS:
            return pd.DataFrame(rows)
        return rows


# --------------------------------------------------------------------------
# Parsing de la sortie NOMAD (format DISPLAY_STATS "bbe ( sol ) obj")
# --------------------------------------------------------------------------

# Exemple de ligne produite par NOMAD avec DISPLAY_STATS "bbe ( sol ) obj" :
#   12 ( 0.123456 -1.234567 0.000001 ... )    3.456789e+00
_STAT_LINE_RE = re.compile(
    r"^\s*(\d+)\s*\(\s*([^)]*?)\s*\)\s+([-+]?\d[\d.eE+\-]*)\s*$"
)


def parse_nomad_stats(stdout_text: str) -> List[NomadStatLine]:
    """Extrait les lignes de statistiques NOMAD (format bbe / sol / obj)
    d'une sortie stdout brute. Les lignes qui ne correspondent pas au motif
    (bannieres, avertissements, etc.) sont simplement ignorees."""
    history: List[NomadStatLine] = []
    for line in stdout_text.splitlines():
        m = _STAT_LINE_RE.match(line)
        if not m:
            continue
        bbe = int(m.group(1))
        x_str = m.group(2).strip()
        try:
            x = [float(v) for v in x_str.split()]
            f = float(m.group(3))
        except ValueError:
            continue
        history.append(NomadStatLine(bbe=bbe, x=x, f=f))
    return history


# --------------------------------------------------------------------------
# Compilation et execution
# --------------------------------------------------------------------------

class NomadTangentRunner:
    """Compile et execute un fichier source NOMAD (type pollTangent.cpp)
    en s'appuyant sur une installation NOMAD 4 deja construite localement.

    Parameters
    ----------
    source_path : str | Path
        Chemin vers le fichier .cpp (ex. pollTangent.cpp).
    nomad_home : str | Path, optional
        Racine du depot NOMAD (contient src/, build/, examples/...).
        Par defaut, lit la variable d'environnement NOMAD_HOME.
    build_subdir : str
        Sous-dossier de build a l'interieur de nomad_home (ex. "build/release").
    exe_name : str
        Nom de l'executable genere.
    work_dir : str | Path, optional
        Dossier de travail pour la compilation/execution (par defaut, le
        dossier du fichier source).
    """

    # Bibliotheques NOMAD 4 typiquement necessaires (le nom exact peut
    # varier selon la version / le systeme -- ajuster au besoin).
    DEFAULT_LIBS = ["nomadAlgos", "nomadUtils", "nomadEval", "sgtelib"]

    def __init__(self, source_path, nomad_home: Optional[str] = None,
                 build_subdir: str = "build/release", exe_name: str = "poll_tangent",
                 work_dir: Optional[str] = None, extra_libs: Optional[Sequence[str]] = None):
        self.source_path = Path(source_path).resolve()
        if not self.source_path.exists():
            raise FileNotFoundError(f"Fichier source introuvable : {self.source_path}")

        self.nomad_home = Path(nomad_home or os.environ.get("NOMAD_HOME", "")).expanduser()
        self.build_dir = self.nomad_home / build_subdir
        self.exe_name = exe_name
        self.work_dir = Path(work_dir or self.source_path.parent).resolve()
        self.libs = list(self.DEFAULT_LIBS) + list(extra_libs or [])

        self.exe_path: Optional[Path] = None

    # ------------------------------------------------------------------ #
    def _include_dirs(self) -> List[Path]:
        # Le repertoire src/ de NOMAD contient les en-tetes (Nomad/nomad.hpp,
        # Algos/..., Math/..., etc.)
        return [self.nomad_home / "src"]

    def _lib_dirs(self) -> List[Path]:
        # Selon la version/plateforme, les .so/.dylib/.lib sont generes
        # dans build_dir/lib ou directement dans build_dir.
        candidates = [self.build_dir / "lib", self.build_dir]
        return [c for c in candidates if c.exists()] or candidates

    # ------------------------------------------------------------------ #
    def compile(self, cxx: str = "g++", std: str = "c++17",
                extra_flags: Optional[Sequence[str]] = None, verbose: bool = True) -> Path:
        """Compile le fichier source en liant contre la bibliotheque NOMAD.
        Retourne le chemin de l'executable genere."""
        if not self.nomad_home.exists():
            raise FileNotFoundError(
                f"NOMAD_HOME introuvable ou invalide : '{self.nomad_home}'. "
                "Definis la variable d'environnement NOMAD_HOME ou passe "
                "nomad_home=... au constructeur."
            )

        if shutil.which(cxx) is None:
            raise RuntimeError(f"Compilateur '{cxx}' introuvable dans le PATH.")

        exe_path = self.work_dir / self.exe_name
        cmd = [cxx, f"-std={std}", "-O2", str(self.source_path), "-o", str(exe_path)]

        for inc in self._include_dirs():
            cmd += ["-I", str(inc)]
        for libdir in self._lib_dirs():
            cmd += ["-L", str(libdir)]
        for lib in self.libs:
            cmd += [f"-l{lib}"]
        # rpath pour que l'executable retrouve les .so a l'execution
        for libdir in self._lib_dirs():
            cmd += [f"-Wl,-rpath,{libdir}"]

        if extra_flags:
            cmd += list(extra_flags)

        if verbose:
            print("Commande de compilation :")
            print(" ", " ".join(cmd))

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            raise RuntimeError(
                f"Echec de compilation (code {proc.returncode}). "
                "Verifie NOMAD_HOME / build_subdir / noms des bibliotheques "
                "(DEFAULT_LIBS peut differer selon ta version de NOMAD)."
            )

        self.exe_path = exe_path
        if verbose:
            print(f"Compilation reussie : {exe_path}")
        return exe_path

    # ------------------------------------------------------------------ #
    def run(self, args: Optional[Sequence[str]] = None, timeout: Optional[float] = None,
            env: Optional[dict] = None) -> NomadTangentResult:
        """Execute l'executable compile et parse sa sortie."""
        if self.exe_path is None or not self.exe_path.exists():
            raise RuntimeError("Executable non compile. Appelle .compile() d'abord.")

        run_env = os.environ.copy()
        for libdir in self._lib_dirs():
            # au cas ou le rpath n'aurait pas suffi (ex. sur certains Linux)
            run_env["LD_LIBRARY_PATH"] = f"{libdir}:{run_env.get('LD_LIBRARY_PATH', '')}"
            run_env["DYLD_LIBRARY_PATH"] = f"{libdir}:{run_env.get('DYLD_LIBRARY_PATH', '')}"
        if env:
            run_env.update(env)

        cmd = [str(self.exe_path)] + list(args or [])
        proc = subprocess.run(cmd, capture_output=True, text=True,
                               cwd=str(self.work_dir), timeout=timeout, env=run_env)

        history = parse_nomad_stats(proc.stdout)
        result = NomadTangentResult(
            history=history,
            raw_stdout=proc.stdout,
            raw_stderr=proc.stderr,
            returncode=proc.returncode,
        )
        if history:
            best = min(history, key=lambda h: h.f)
            result.x_best = best.x
            result.f_best = best.f
            result.nb_bb_evals = history[-1].bbe

        if proc.returncode != 0:
            print("[nomad_tangent_wrapper] Attention : code de retour non nul.")
            print(proc.stderr)

        return result


# --------------------------------------------------------------------------
# Auto-test du parsing (ne necessite pas d'installation NOMAD)
# --------------------------------------------------------------------------

def _selftest_parse():
    sample_stdout = """\
NOMAD - version 4.4.0
Some banner text that should be ignored
1 ( 0.100000 0.200000 0.300000 0.400000 0.500000 0.600000 )    1.234500e+01
7 ( 0.050000 0.180000 0.290000 0.410000 0.480000 0.590000 )    9.876000e+00
23 ( 0.010000 0.150000 0.250000 0.400000 0.450000 0.550000 )    3.210000e+00
Optimization done, best feasible solution found.
"""
    history = parse_nomad_stats(sample_stdout)
    assert len(history) == 3, f"Attendu 3 lignes, obtenu {len(history)}"
    assert history[0].bbe == 1
    assert history[-1].f == 3.21
    assert len(history[0].x) == 6
    print("Auto-test du parsing : OK")
    print(history)


if __name__ == "__main__":
    _selftest_parse()

    print(
        "\nExemple d'utilisation complete (necessite une installation NOMAD "
        "locale compilee) :\n"
        "\n"
        "    runner = NomadTangentRunner(\n"
        "        source_path='pollTangent.cpp',\n"
        "        nomad_home='/chemin/vers/nomad',\n"
        "    )\n"
        "    runner.compile()\n"
        "    result = runner.run()\n"
        "    print(result.f_best, result.x_best, result.nb_bb_evals)\n"
        "    df = result.history_dataframe()\n"
    )
