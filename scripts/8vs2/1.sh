base=experiments/8vs2

find $base/balanced_cl -type f -iname "*FR*"  -exec bash {} \; 
find $base/unbalanced_cl -type f -iname "*FR*"  -exec bash {} \; 
# find experiments/${OLD_VS_NEW} -type f -iname "*_FR_*5em3*"  -exec bash {} \; 
