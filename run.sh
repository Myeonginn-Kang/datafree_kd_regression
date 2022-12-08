for i in bikesharing compactiv cpusmall ctscan indoorloc mv pole puma32h telemonitoirng
do
  for j in proposed sampling DI
  do
    for k in 25 50 100
    do
      for l in 0 1 2 3 4 5 6 7 8 9
      do
        python main.py $i $l $k $j
      done
    done
  done
done
    