# Make for arm64 Macs
arm: env_helpers.c
	arch -x86_64 gcc -shared -o env_helpers.so -fPIC env_helpers.c -lm -O2

# Make for non-arm64 Macs
non-arm: env_helpers.c
	gcc -shared -o env_helpers.so -fPIC env_helpers.c -lm -O2

leak-test:
	make arm && gcc -o c_test ./unit_tests/c_test.c env_helpers.c -lm && leaks -atExit -- ./c_test

q-learning:
	gcc -o q_learning Q-learning.c env_helpers.c hashmap.c -lm && leaks -atExit -- ./q_learning