cd /gpfs/home/cx2484/convexe
module load gcc/13.4.0
module load cmake/4.4.0

# 1. Localiser l'exécutable
find build -name "projectedPollMethod.exe"

# 2. Se placer dans son dossier et le lancer
cd <dossier_trouvé_par_find>
./projectedPollMethod.exe