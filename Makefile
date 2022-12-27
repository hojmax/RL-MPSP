# Make for arm64 Macs
arm: env_helpers.c
	arch -x86_64 gcc -shared -o env_helpers.so -fPIC env_helpers.c -lm

# Make for non-arm64 Macs
non-arm: env_helpers.c
	gcc -shared -o env_helpers.so -fPIC env_helpers.c -lm