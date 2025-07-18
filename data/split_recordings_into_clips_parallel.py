import os
import av
import argparse
from tqdm import tqdm
from utils.files import find_files_recursively
from utils.logger import setup_basic_logger
from threading import Thread, Lock
from data.split_utils import is_blank_frame, try_read_qr_code, open_output_writer


log = setup_basic_logger(os.path.basename(__file__))


def worker(
    worker_name: str,
    fragments_filepaths: list[str],
    output_dir: str,
    shared_set: set[str],
    lock: Lock,
    pbar: tqdm) -> None:

    current_item_id = None
    current_modifiers = None

    is_recording = False

    output_container = None
    output_stream = None
    current_start_pts = None

    for fragment_filepath in tqdm(fragments_filepaths, desc=f"[{worker_name}] Processing fragments", unit="fragment", disable=True):

        # Open the video file
        with av.open(fragment_filepath) as input_container:
            input_stream = input_container.streams.video[0]

            # Decode input fragment frame by frame
            for frame_idx, input_frame in enumerate(input_container.decode(video=0)):

                # Convert frame to RGB
                frame_rgb = input_frame.to_ndarray(format="rgb24")

                # Skip over black frames
                if is_blank_frame(frame_rgb):
                    continue

                # Check if the frame is a separator frame
                metadata = try_read_qr_code(frame_rgb)

                if metadata is not None and isinstance(metadata, dict) and "item_id" in metadata:
                    # It is a metadata frame

                    # If we are currently recording, stop the recording
                    if is_recording:
                        # Flush and close current video
                        for packet in output_stream.encode():
                            packet.time_base = input_stream.time_base
                            output_container.mux(packet)
                        output_container.close()
                        is_recording = False

                    if (current_item_id == metadata["item_id"]) and (current_modifiers == metadata["modifiers"]):
                        # We already saw this metadata frame
                        continue

                    # Decode QR code
                    current_item_id = metadata["item_id"]
                    current_modifiers = metadata["modifiers"]

                    # Check the set
                    video_id = f"{fragment_filepath}_{current_item_id}_{current_modifiers}"
                    with lock:
                        if video_id in shared_set:
                            # Another thread already started processing that video. This thread's job is done.
                            log.info(f"[Thread {worker_name}] Video {video_id} already processed.]")
                            return

                        else:
                            shared_set.add(video_id)
                            pbar.update(1)

                    # Continue to next frame
                    continue

                else:
                    # This is not a metadata frame

                    # If we don't have any metadata yet, we will have to skip this frame
                    if current_item_id is None:
                        # log.info(f"[{worker_name}] Need to skip frame {frame_idx} because we haven't seen a metadata frame yet.")
                        continue

                    # First frame of a new sequence
                    if not is_recording:
                        # Create new output writer
                        output_filename = f"{current_item_id}_{current_modifiers}.mp4"
                        output_filepath = os.path.join(output_dir, output_filename)
                        output_container, output_stream = open_output_writer(output_filepath, input_stream)
                        is_recording = True
                        current_start_pts = input_frame.pts
                        log.info(f"[Thread {worker_name}] Starting new clip (item id: {current_item_id}, modifiers: {current_modifiers}). Output filepath is \"{output_filepath}\".")

                    # Create a new frame with the pixel values of the input frame
                    output_frame = av.VideoFrame.from_ndarray(input_frame.to_ndarray(), format=input_frame.format.name)

                    # Calculate new PTS relative to the first frame of the clip
                    rel_pts = input_frame.pts - current_start_pts
                    output_frame.pts = rel_pts

                    # Encode and write to output
                    try:
                        # Encode and mux
                        for packet in output_stream.encode(output_frame):
                            packet.time_base = input_stream.time_base
                            output_container.mux(packet)

                    except Exception as e:
                        log.exception(f"[Thread {worker_name}] Error encoding frame {frame_idx} of fragment {fragment_filepath} to {output_filepath}: {e}")

                        # Stop recording
                        current_item_id = None
                        current_modifiers = None
                        is_recording = False

                        # This recording is burnt
                        if os.path.exists(output_filepath):
                            os.remove(output_filepath)

                        # Skip ahead until we find the next metadata frame

    # Close the last output file
    if is_recording:
        # Flush residue from this fragment
        for packet in output_stream.encode():
            packet.time_base = input_stream.time_base
            output_container.mux(packet)

        output_container.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate recorded frames")
    parser.add_argument("--input_dir", type=str, help="Directory where to search for fragments", default="/tmp/66e4fbe5-7170-4d61-8585-90137782784a")
    parser.add_argument("--output_dir", type=str, help="Directory where to save the output fragments", default="/tmp")
    parser.add_argument("--num_workers", type=int, help="Number of workers", default=8)
    args = vars(parser.parse_args())

    fragments_filepaths = sorted(list(find_files_recursively(args["input_dir"], file_extensions=[".mkv"])))

    shared_set: set[str] = set()
    lock = Lock()

    with tqdm(total=400, desc="Video clips") as pbar:
        threads: list[Thread] = []

        for i in range(args["num_workers"]):
            offset = len(fragments_filepaths) // args["num_workers"] * i
            worker_fragments = fragments_filepaths[offset:]
            t = Thread(
                target=worker,
                args=(str(i), worker_fragments, args["output_dir"], shared_set, lock, pbar)
            )
            t.start()
            threads.append(t)

        # Wait for all workers to finish
        for t in threads:
            t.join()

    print(f"Length of final set: {len(shared_set)}")
