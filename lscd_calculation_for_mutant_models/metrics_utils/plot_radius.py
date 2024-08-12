import matplotlib.pyplot as plt
import numpy as np


def plot_radius_values(
    radius_ip, x_threshold, selected_radius, x_threshold_elbow, selected_radius_elbow, poly, output_path, i
):
    x, y = list(range(len(radius_ip))), sorted(radius_ip)
    plt.clf()

    # plt.annotate(f"Hand-picked\nR={round(selected_radius, 2)}", (x_threshold, selected_radius),
    #         ( x_threshold - min(150, round(0.20 * len(x))),  # x coord
    #         selected_radius + min(150, round(0.13 * len(y)))),
    #     bbox=dict(boxstyle="round", fc="tab:green"))

    # plt.plot([x_threshold, x_threshold], [0, selected_radius],color='tab:green', linestyle='dashed')
    # plt.plot([0, x_threshold], [selected_radius, selected_radius], color='tab:green', linestyle='dashed')

    # Plot the fitted polynomial over the distribution
    # plt.plot(np.polyval(poly, y), 'k-')
    poly[1].sort()
    plt.plot(poly[1], x, "k-")
    # plt.plot(y, x)
    # plt.plot(x, y_1)
    # plt.xlabel("# Train Data Points")
    # plt.ylabel("Radius")

    plt.xlabel("Radius", fontsize=18)
    plt.ylabel("# Train Data Points", fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Plot the annotation for the radius - data v/s radius
    # plt.annotate(f"Selected \nR = {round(selected_radius_elbow, 3)}", (x_threshold_elbow, selected_radius_elbow),
    #             (x_threshold_elbow - max(0, 0.2 * len(x)),  # x coord
    #             selected_radius_elbow + 0.2),
    #             bbox=dict(boxstyle="round", fc="skyblue"),
    #             arrowprops=dict(arrowstyle="->",
    #             connectionstyle="arc3"))  # aqua

    # plt.plot(x_threshold_elbow, selected_radius_elbow - 0.036, 'o', color= 'skyblue', markeredgecolor = 'black', markersize=10)
    # plt.plot([x_threshold_elbow, x_threshold_elbow], [0, selected_radius_elbow- 0.036],color='skyblue', linestyle='dashed')
    # plt.plot([0, x_threshold_elbow], [selected_radius_elbow - 0.036, selected_radius_elbow - 0.036], color='skyblue', linestyle='dashed')

    # Plot the annotation for the radius - radius v/s data
    plt.annotate(
        f"Selected \nr = {round(selected_radius_elbow, 3)}",
        (selected_radius_elbow - 0.05, x_threshold_elbow),
        (selected_radius_elbow, x_threshold_elbow),
        bbox=dict(boxstyle="round", fc="skyblue"),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        fontsize=15,
    )  # aqua

    plt.plot(selected_radius_elbow, x_threshold_elbow, "o", color="skyblue", markeredgecolor="black", markersize=10)
    plt.plot(
        [selected_radius_elbow, selected_radius_elbow], [0, x_threshold_elbow], color="skyblue", linestyle="dashed"
    )
    plt.plot([0, selected_radius_elbow], [x_threshold_elbow, x_threshold_elbow], color="skyblue", linestyle="dashed")

    # plt.gca().invert_xaxis()  # changes the scale
    # plt.gca().invert_yaxis()

    plt.title("Radius Selection \n GTSRB (Class: " + str(i) + ")", fontsize=21)

    # plt.legend(fontsize=21, loc='upper right')
    plt.tight_layout()
    # plt.gca().set_aspect('equal', 'datalim')

    plt.savefig(output_path, dpi=600)
