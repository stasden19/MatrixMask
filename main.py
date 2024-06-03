import argparse
import cv2


def main(video, matrix, RES=(1280, 720)):
    main_capture = cv2.VideoCapture(video)
    matrix_capture = cv2.VideoCapture(matrix)
    substractor = cv2.createBackgroundSubtractorKNN()
    while True:
        try:
            frame = main_capture.read()[1]
            frame = cv2.resize(frame, RES, interpolation=cv2.INTER_AREA)

            bg_frame = matrix_capture.read()[1]
            bg_frame = cv2.resize(bg_frame, RES, interpolation=cv2.INTER_AREA)

            mask = substractor.apply(frame, 1)
            bitwise = cv2.bitwise_and(bg_frame, bg_frame, mask=mask)
        except :
            pass
        cv2.imshow('', bitwise)
        if cv2.waitKey(1) == 27:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Наложение маски на видео')
    parser.add_argument('-v', '--video', help='Главный поток видео')
    parser.add_argument('-m', '--matrix', help='Видео с матрицей или чем-то другим')
    args = parser.parse_args()
    main(args.video, args.matrix)
