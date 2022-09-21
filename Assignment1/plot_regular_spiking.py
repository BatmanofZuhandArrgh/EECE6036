from spike import plot_membrane_potential

def main():
    a=0.02; b=0.25; c=-65;  d=6

    plot_membrane_potential(
        num_trials=41,
        folder_name='regular_spiking',
        a=a,
        b=b,
        c=c,
        d=d
    )

if __name__ == '__main__':
    main()