

IMAGE_NAME := pcuda

$(IMAGE_NAME): Dockerfile
	docker build -t $(IMAGE_NAME) -f $^ .

sh:
	docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
		-v $(shell pwd):/app \
		--privileged \
		--name $(IMAGE_NAME) \
		$(IMAGE_NAME)

