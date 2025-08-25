#include "filter_impl.h"

#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include "logo.h"

struct rgb {
    uint8_t r, g, b;
};

struct PixelState {
    float bg_r, bg_g, bg_b;
    int time;
};

std::vector<PixelState> pixel_states;

float colorDistance(const rgb& p1, const rgb& p2) {
    return std::sqrt((p1.r - p2.r) * (p1.r - p2.r) +
                     (p1.g - p2.g) * (p1.g - p2.g) +
                     (p1.b - p2.b) * (p1.b - p2.b));
}

void applyMorphologicalOpening(std::vector<float>& motion_mask, int width, int height, int radius) {
    std::vector<float> temp_mask = motion_mask;

    // Apply erosion
    for (int y = radius; y < height - radius; ++y) {
        for (int x = radius; x < width - radius; ++x) {
            float min_val = 255.0f;
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dx * dx + dy * dy <= radius * radius) {
                        min_val = std::min(min_val, temp_mask[(y + dy) * width + (x + dx)]);
                    }
                }
            }
            motion_mask[y * width + x] = min_val;
        }
    }

    // Apply dilation
    temp_mask = motion_mask;
    for (int y = radius; y < height - radius; ++y) {
        for (int x = radius; x < width - radius; ++x) {
            float max_val = 0.0f;
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dx * dx + dy * dy <= radius * radius) {
                        max_val = std::max(max_val, temp_mask[(y + dy) * width + (x + dx)]);
                    }
                }
            }
            motion_mask[y * width + x] = max_val;
        }
    }
}

void applyHysteresisThresholding(std::vector<float>& motion_mask, int width, int height, float low_thresh, float high_thresh) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float& value = motion_mask[y * width + x];
            if (value >= high_thresh) {
                value = 1.0f;  // Strong motion
            } else if (value < low_thresh) {
                value = 0.0f;  // No motion
            } else {
                // Weak motion: check neighbors
                bool strong_neighbor = false;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int ny = y + dy;
                        int nx = x + dx;
                        if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                            if (motion_mask[ny * width + nx] >= high_thresh) {
                                strong_neighbor = true;
                                break;
                            }
                        }
                    }
                    if (strong_neighbor) break;
                }
                value = strong_neighbor ? 1.0f : 0.0f;
            }
        }
    }
}

std::vector<float> motion_mask;

extern "C" {
    void filter_impl(uint8_t* buffer, int width, int height, int stride, int pixel_stride)
    {
        if (pixel_states.empty()) {
            pixel_states.resize(width * height);
            for (auto& state : pixel_states) {
                state.bg_r = state.bg_g = state.bg_b = 0.0f;
                state.time = 0;
            }

            for (int y = 0; y < height; ++y) {
                rgb* lineptr = (rgb*)(buffer + y * stride);
                for (int x = 0; x < width; ++x) {
                    int idx = y * width + x;
                    PixelState& state = pixel_states[idx];
                    rgb& pixel = lineptr[x];
                    state.bg_r = pixel.r;
                    state.bg_g = pixel.g;
                    state.bg_b = pixel.b;
                }
            }
            return;
        }

        if (motion_mask.empty()) {
            motion_mask.resize(width * height, 0.0f);
        }

        for (int y = 0; y < height; ++y) {
            rgb* lineptr = (rgb*)(buffer + y * stride);
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                PixelState& state = pixel_states[idx];

                rgb& pixel = lineptr[x];
                float distance = colorDistance(pixel, {uint8_t(state.bg_r), uint8_t(state.bg_g), uint8_t(state.bg_b)});

                if (distance < 25.0f) {
                    state.time = 0;
                } else {
                    motion_mask[idx] = distance;
                    state.time++;

                    if (state.time > 100) {
                        state.bg_r = pixel.r;
                        state.bg_g = pixel.g;
                        state.bg_b = pixel.b;
                        state.time = 0;
                    }
                }
            }
        }

        int radius = std::max(3, std::min(width, height) / 100);
        applyMorphologicalOpening(motion_mask, width, height, radius);

        float low_thresh = 4.0f;
        float high_thresh = 30.0f;
        applyHysteresisThresholding(motion_mask, width, height, low_thresh, high_thresh);

        for (int y = 0; y < height; ++y) {
            rgb* lineptr = (rgb*)(buffer + y * stride);
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                if (motion_mask[idx] > 0.0f) {
                    lineptr[x].r = std::min(255, lineptr[x].r + uint8_t(0.5f * 255));
                }
            }
        }
    }   
}
