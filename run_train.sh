# ../gfn-env/bin/python train.py \
#     env=tetris proxy=tetris \
#     env.width=5 env.height=10 \
#     gflownet=trajectorybalance loss=trajectorybalance \
#     gflownet.optimizer.n_train_steps=3000 \
#     device=cuda \
#     logger.do.online=True logger.debug=True \
#     evaluator.top_k=30 evaluator.n_top_k=1000 \
#     buffer.test.type=random buffer.test.n=10


# ../gfn-env/bin/python train.py \
#     env=tetris proxy=tetris \
#     gflownet.optimizer.n_train_steps=1 \
#     device=cuda \
#     logger.do.online=False logger.debug=True \
#     evaluator.top_k=50 evaluator.n_top_k=2000


../gfn-env/bin/python train.py env=tetris proxy=tetris gflownet=trajectorybalance \
  env.width=7 env.height=9 'env.pieces=[I,J,L,O,S,T]' 'env.rotations=[0,90]' \
  env.allow_eos_before_full=True \
  n_samples=100 \
  proxy.reward_function=exponential '+proxy.reward_function_kwargs.alpha=1.0' '+proxy.reward_function_kwargs={beta: 10.0}' \
  gflownet.optimizer.n_train_steps=4000 \
  evaluator.period=3000 \
  device=cuda logger.do.online=False \
  user.logdir.root=./logs
  # hydra.run.dir=./logs/tetris_tb_sharp2