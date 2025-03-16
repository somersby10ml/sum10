import argparse
import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# 이전에 작성한 모듈 임포트
from lecagy.apple_game_dqn import (AppleGameAgent, AppleGameDQN, AppleGameEnv,
                                   ReplayBuffer)
from lecagy.board_generator import (generate_dataset, verify_board_solvability,
                                    visualize_board)


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='Apple Game DQN Training')

    # 학습 파라미터
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float,
                        default=0.99, help='Discount factor')
    parser.add_argument('--eps_start', type=float, default=1.0,
                        help='Starting epsilon for exploration')
    parser.add_argument('--eps_end', type=float, default=0.05,
                        help='Ending epsilon for exploration')
    parser.add_argument('--eps_decay', type=float,
                        default=0.995, help='Epsilon decay rate')

    # 데이터 파라미터
    parser.add_argument('--num_boards', type=int, default=200,
                        help='Number of boards to generate')
    parser.add_argument('--min_regions', type=int, default=25,
                        help='Minimum regions per board')
    parser.add_argument('--use_existing', action='store_true',
                        help='Use existing dataset if available')
    parser.add_argument('--dataset_path', type=str,
                        default='apple_game_dataset.pkl', help='Path to save/load dataset')

    # 학습 구성
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train over the dataset')
    parser.add_argument('--episodes_per_board', type=int,
                        default=10, help='Episodes to train on each board')
    parser.add_argument('--eval_freq', type=int, default=100,
                        help='Frequency of evaluation during training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available()
                        else 'cpu', help='Device to train on')

    # 모델 파라미터
    parser.add_argument('--save_dir', type=str,
                        default='models', help='Directory to save models')
    parser.add_argument('--model_name', type=str,
                        default='apple_game_dqn', help='Base name for saved models')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to load a pre-trained model')

    return parser.parse_args()


def setup_training(args):
    """학습 환경 설정"""
    # 저장 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)

    # 환경 및 에이전트 초기화
    env = AppleGameEnv(height=13, width=10)
    agent = AppleGameAgent(
        board_height=13,
        board_width=10,
        learning_rate=args.lr
    )

    # 학습 파라미터 설정
    agent.batch_size = args.batch_size
    agent.gamma = args.gamma
    agent.epsilon = args.eps_start
    agent.epsilon_min = args.eps_end
    agent.epsilon_decay = args.eps_decay

    # 기존 모델 로드 (있는 경우)
    if args.load_model and os.path.isfile(args.load_model):
        agent.load_model(args.load_model)
        print(f"Loaded pre-trained model from {args.load_model}")

    return env, agent


def prepare_dataset(args):
    """학습 데이터셋 준비"""
    # 기존 데이터셋 사용 또는 새로 생성
    if args.use_existing and os.path.isfile(args.dataset_path):
        with open(args.dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        print(
            f"Loaded existing dataset with {len(dataset)} boards from {args.dataset_path}")
    else:
        print(f"Generating new dataset with {args.num_boards} boards...")
        dataset = generate_dataset(
            num_boards=args.num_boards,
            min_regions=args.min_regions,
            save_path=args.dataset_path
        )
        print(f"Generated and saved dataset with {len(dataset)} boards")

    return dataset


def train_epoch(agent, env, dataset, args, epoch):
    """한 에포크 학습"""
    print(f"\nTraining Epoch {epoch+1}/{args.epochs}")
    total_episodes = len(dataset) * args.episodes_per_board

    all_rewards = []
    all_scores = []
    all_losses = []

    episode_counter = 0

    # 각 보드에 대해 학습
    for board_idx, (board, regions) in enumerate(tqdm(dataset, desc=f"Epoch {epoch+1} Progress")):
        for episode in range(args.episodes_per_board):
            # 보드 초기화
            state = env.reset(board)
            done = False
            episode_reward = 0

            # 에피소드 플레이
            while not done:
                # 행동 선택
                action = agent.select_action(state)

                # 환경에서 한 스텝 진행
                next_state, reward, done, info = env.step(action)

                # 경험 저장
                agent.memory.add(state, action, reward, next_state, done)

                # 모델 최적화
                loss = agent.optimize_model()
                if loss:
                    all_losses.append(loss)

                # 상태 업데이트
                state = next_state
                episode_reward += reward

            # 에피소드 결과 기록
            all_rewards.append(episode_reward)
            all_scores.append(env.score)

            episode_counter += 1

            # 주기적으로 성능 평가 및 모델 저장
            if episode_counter % args.eval_freq == 0:
                avg_reward = np.mean(all_rewards[-args.eval_freq:])
                avg_score = np.mean(all_scores[-args.eval_freq:])
                avg_loss = np.mean(
                    all_losses[-min(len(all_losses), args.eval_freq):]) if all_losses else 0

                print(f"\nProgress: {episode_counter}/{total_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, Avg Score: {avg_score:.2f}, "
                      f"Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f}")

                # 모델 저장
                model_path = os.path.join(
                    args.save_dir, f"{args.model_name}_epoch{epoch+1}_ep{episode_counter}.pth")
                agent.save_model(model_path)

    # 에포크 결과
    epoch_stats = {
        'rewards': all_rewards,
        'scores': all_scores,
        'losses': all_losses if all_losses else [0],
        'final_epsilon': agent.epsilon
    }

    return epoch_stats


def evaluate_model(agent, env, test_dataset, num_episodes=50):
    """모델 평가"""
    print("\nEvaluating model...")

    success_count = 0
    total_score = 0
    total_moves = 0

    # 테스트할 보드 선택
    if len(test_dataset) > num_episodes:
        test_boards = random.sample(test_dataset, num_episodes)
    else:
        test_boards = test_dataset

    for board, regions in tqdm(test_boards, desc="Evaluation"):
        # 보드 초기화
        state = env.reset(board)
        done = False

        # 평가 시에는 탐험하지 않음
        epsilon_backup = agent.epsilon
        agent.epsilon = 0

        # 에피소드 플레이
        while not done and env.moves < 100:  # 최대 100번 이동 제한
            action = agent.select_action(state)
            state, _, done, _ = env.step(action)

        # 결과 기록
        total_score += env.score
        total_moves += env.moves

        # 성공 여부 확인 (모든 사과 제거)
        if np.all(env.board == 0):
            success_count += 1

        # 엡실론 복원
        agent.epsilon = epsilon_backup

    # 결과 정리
    success_rate = success_count / len(test_boards)
    avg_score = total_score / len(test_boards)
    avg_moves = total_moves / len(test_boards)

    print(f"Evaluation Results:")
    print(
        f"Success Rate: {success_rate:.4f} ({success_count}/{len(test_boards)})")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Average Moves: {avg_moves:.2f}")

    return success_rate, avg_score, avg_moves


def plot_training_progress(epoch_stats_list, save_path=None):
    """학습 진행 상황 시각화"""
    epochs = len(epoch_stats_list)

    # 데이터 준비
    all_rewards = []
    all_scores = []
    all_losses = []
    epsilons = []

    for epoch, stats in enumerate(epoch_stats_list):
        all_rewards.extend(stats['rewards'])
        all_scores.extend(stats['scores'])
        all_losses.extend(stats['losses'])
        epsilons.append(stats['final_epsilon'])

    # 이동 평균 계산
    window_size = min(100, len(all_rewards) // 10)
    if window_size < 1:
        window_size = 1

    reward_ma = np.convolve(all_rewards, np.ones(
        window_size)/window_size, mode='valid')
    score_ma = np.convolve(all_scores, np.ones(
        window_size)/window_size, mode='valid')

    if len(all_losses) > window_size:
        loss_ma = np.convolve(all_losses, np.ones(
            window_size)/window_size, mode='valid')
    else:
        loss_ma = all_losses

    # 그래프 그리기
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))

    # 보상 그래프
    axs[0].plot(all_rewards, 'b-', alpha=0.3)
    axs[0].plot(np.arange(window_size-1, len(all_rewards)), reward_ma, 'b-')
    axs[0].set_title('Episode Rewards')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Reward')
    axs[0].grid(True)

    # 점수 그래프
    axs[1].plot(all_scores, 'g-', alpha=0.3)
    axs[1].plot(np.arange(window_size-1, len(all_scores)), score_ma, 'g-')
    axs[1].set_title('Episode Scores (Removed Regions)')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Score')
    axs[1].grid(True)

    # 손실 그래프
    axs[2].plot(all_losses, 'r-', alpha=0.3)
    if len(loss_ma) > 1:
        axs[2].plot(np.arange(window_size-1, len(all_losses)), loss_ma, 'r-')
    axs[2].set_title('Training Loss')
    axs[2].set_xlabel('Optimization Step')
    axs[2].set_ylabel('Loss')
    axs[2].set_yscale('log')
    axs[2].grid(True)

    plt.tight_layout()

    # 그래프 저장
    if save_path:
        plt.savefig(save_path)
        print(f"Training progress plot saved to {save_path}")

    plt.show()


def interactive_demo(agent, env, dataset):
    """모델 시연"""
    print("\nInteractive Demo")
    print("Select a board to test the agent on:")

    # 보드 목록 표시
    for i in range(min(5, len(dataset))):
        board, regions = dataset[i]
        print(f"{i+1}. Board with {len(regions)} regions")

    choice = input("Enter board number (1-5) or 'r' for random: ")

    if choice.lower() == 'r':
        board_idx = np.random.randint(0, len(dataset))
    else:
        try:
            board_idx = int(choice) - 1
            if board_idx < 0 or board_idx >= len(dataset):
                print("Invalid choice, using random board.")
                board_idx = np.random.randint(0, len(dataset))
        except:
            print("Invalid input, using random board.")
            board_idx = np.random.randint(0, len(dataset))

    # 선택한 보드로 게임 진행
    board, regions = dataset[board_idx]
    state = env.reset(board)

    print("\nInitial Board:")
    env.render()

    # 평가 모드
    epsilon_backup = agent.epsilon
    agent.epsilon = 0

    step = 1
    done = False

    while not done and step <= 100:
        print(f"\nStep {step}:")
        action = agent.select_action(state)
        print(
            f"Agent selects region: Start=({action[0]}, {action[1]}), End=({action[2]}, {action[3]})")

        state, reward, done, _ = env.step(action)
        env.render()

        region_sum = np.sum(
            board[action[0]:action[2]+1, action[1]:action[3]+1])
        if region_sum == 10:
            print(f"Sum of region: {region_sum} = 10, region removed!")
        else:
            print(f"Sum of region: {region_sum} ≠ 10, no removal")

        step += 1

        if done:
            if np.all(env.board == 0):
                print("Game completed successfully! All apples removed.")
            else:
                print("Game ended without removing all apples.")

        # 사용자 계속 진행 확인
        if not done:
            cont = input("Press Enter to continue, 'q' to quit: ")
            if cont.lower() == 'q':
                break

    # 엡실론 복원
    agent.epsilon = epsilon_backup


def main():
    """메인 함수"""
    # 인자 파싱
    args = parse_args()

    # 학습 환경 설정
    env, agent = setup_training(args)

    # 데이터셋 준비
    dataset = prepare_dataset(args)

    # 학습 수행
    epoch_stats_list = []
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_stats = train_epoch(agent, env, dataset, args, epoch)
        epoch_stats_list.append(epoch_stats)

        # 각 에포크마다 모델 저장
        epoch_model_path = os.path.join(
            args.save_dir, f"{args.model_name}_epoch{epoch+1}.pth")
        agent.save_model(epoch_model_path)

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # 학습 결과 시각화
    plot_training_progress(
        epoch_stats_list,
        save_path=os.path.join(
            args.save_dir, f"{args.model_name}_progress.png")
    )

    # 최종 모델 평가
    success_rate, avg_score, avg_moves = evaluate_model(agent, env, dataset)

    # 최종 모델 저장
    final_model_path = os.path.join(
        args.save_dir, f"{args.model_name}_final.pth")
    agent.save_model(final_model_path)

    # 상호작용 데모
    interactive_demo(agent, env, dataset)


if __name__ == "__main__":
    main()
