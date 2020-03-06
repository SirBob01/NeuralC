COMPILE_FILES="";
for i in ../src/*.c;
do
    COMPILE_FILES=$COMPILE_FILES$" "$i;
done
gcc -g xor.c -lm $COMPILE_FILES -o xor
gcc -g sin_function.c -lm $COMPILE_FILES -o sin_function