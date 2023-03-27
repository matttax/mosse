from utils import *
import cv2
import numpy as np
from itertools import count


class MOSSE:
    def __init__(self, video_path, lr=0.125, sigma=100, num_pretrain=1, rotate=False):
        self.lr = lr
        self.sigma = sigma
        self.num_pretrain = num_pretrain
        self.rotate = rotate
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)

    def start_tracking(self):
        # Получаем от пользователя размер изображения
        ret, init_img = self.cap.read()
        if not ret:
            print("error")
            cv2.waitKey(0)
            return
        frame_image, ground_truth = self._get_frame(init_img)

        # Применяем фильтр Гаусса
        response_map = self._get_gauss_response(frame_image, ground_truth)

        # Выполняем Фурье-преобразование
        g = response_map[ground_truth[1]:ground_truth[1] + ground_truth[3], ground_truth[0]:ground_truth[0] + ground_truth[2]]
        G = np.fft.fft2(g)

        # Инициализируем матрицы A и B для поиска частот фильтра
        F = frame_image[ground_truth[1]:ground_truth[1] + ground_truth[3], ground_truth[0]:ground_truth[0] + ground_truth[2]]
        A, B = self._pre_training(F, G)

        for i in count(0):
            ret, current_frame = self.cap.read()
            if not ret:
                break

            current_frame = cv2.resize(current_frame, (640, 480), interpolation=cv2.INTER_AREA)
            frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            frame_gray = frame_gray.astype(np.float32)
            if i == 0:
                A = self.lr * A
                B = self.lr * B
                pos = ground_truth.copy()
                clip_pos = np.array([pos[0], pos[1], pos[0] + pos[2], pos[1] + pos[3]]).astype(np.int64)
            else:
                # Вычисляем частоты фильтра
                W = A / B
                F = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                F = pre_process(cv2.resize(F, (ground_truth[2], ground_truth[3])))

                # Вычисляем отклик в частотной области
                Gi = W * np.fft.fft2(F)
                gi = linear_mapping(np.fft.ifft2(Gi))
                filter = np.abs(255 * linear_mapping(np.fft.ifft2(W))).astype(np.uint8)
                result = np.abs(255 * gi).astype(np.uint8)

                # Ищем максимум
                max_value = np.max(gi)
                max_pos = np.where(gi == max_value)
                dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
                dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)

                # Обновляем положение окна
                pos[0] = pos[0] + dx
                pos[1] = pos[1] + dy
                clip_pos[0] = np.clip(pos[0], 0, current_frame.shape[1])
                clip_pos[1] = np.clip(pos[1], 0, current_frame.shape[0])
                clip_pos[2] = np.clip(pos[0] + pos[2], 0, current_frame.shape[1])
                clip_pos[3] = np.clip(pos[1] + pos[3], 0, current_frame.shape[0])
                clip_pos = clip_pos.astype(np.int64)

                # Получаем новое окно
                F = frame_gray[clip_pos[1]:clip_pos[3], clip_pos[0]:clip_pos[2]]
                F = pre_process(cv2.resize(F, (ground_truth[2], ground_truth[3])))

                # Обновляем матрицы
                A = self.lr * (G * np.conjugate(np.fft.fft2(F))) + (1 - self.lr) * A
                B = self.lr * (np.fft.fft2(F) * np.conjugate(np.fft.fft2(F))) + (1 - self.lr) * B

            # Визаулизация
            cv2.rectangle(current_frame, (pos[0], pos[1]), (pos[0] + pos[2], pos[1] + pos[3]), (255, 0, 0), 2)
            if i == 0:
                cv2.imshow('demo', current_frame)
            else:
                # Вставляем изображения фильтра и Гауссовского отклика
                in_frame = current_frame.copy()
                hi_b = cv2.cvtColor(filter, cv2.COLOR_GRAY2BGR)
                gi_b = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                rect_up = in_frame.shape[0] - hi_b.shape[0]
                rect_down = in_frame.shape[0]
                rect_right = hi_b.shape[1]
                rect_left = 0
                in_frame[rect_up:rect_down, rect_left:rect_right] = hi_b
                in_frame[rect_up:rect_down, rect_right:(rect_right * 2)] = gi_b
                cv2.imshow('demo', in_frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        self.cap.release()
        return

    def _get_frame(self, init_img):
        init_img = cv2.resize(init_img, (640, 480), interpolation=cv2.INTER_AREA)
        frame_image = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)
        frame_image = frame_image.astype(np.float32)
        ground_truth = cv2.selectROI('demo', init_img, False, False)
        ground_truth = np.array(ground_truth).astype(np.int64)
        return frame_image, ground_truth

    # Инициализация матриц A и B на базовом кадре
    def _pre_training(self, init_frame, G):
        height, width = G.shape
        F = cv2.resize(init_frame, (width, height))
        # Препроцессинг
        F = pre_process(F)
        Ai = G * np.conjugate(np.fft.fft2(F))
        Bi = np.fft.fft2(init_frame) * np.conjugate(np.fft.fft2(init_frame))
        for _ in range(self.num_pretrain):
            if self.rotate:
                F = pre_process(random_warp(init_frame))
            else:
                F = pre_process(init_frame)
            Ai = Ai + G * np.conjugate(np.fft.fft2(F))
            Bi = Bi + np.fft.fft2(F) * np.conjugate(np.fft.fft2(F))
        return Ai, Bi

    # Гауссовский отклик
    def _get_gauss_response(self, img, gt):
        height, width = img.shape
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        # Вычисляем центр объекта
        center_x = gt[0] + 0.5 * gt[2]
        center_y = gt[1] + 0.5 * gt[3]
        dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * self.sigma)
        # Получаем отклик
        response = np.exp(-dist)
        # Нормализуем
        response = linear_mapping(response)
        return response


img_path = 'amoeba.mp4'
tracker = MOSSE(img_path)
tracker.start_tracking()
cv2.destroyAllWindows()
