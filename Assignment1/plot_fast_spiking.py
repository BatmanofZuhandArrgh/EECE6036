from spike import plot_membrane_potential

def main():
    a=0.1; b=0.2; c=-65;  d=2

    plot_membrane_potential(
        num_trials=41,
        folder_name='fast_spiking',
        a=a,
        b=b,
        c=c,
        d=d,
        compare_array='Assignment1/regular_spiking/RS.npy'
    )

if __name__ == '__main__':
    main()