from imaging.operations import Operation

##############################################################################
class FieldInhomogeneityCorrection (Operation):

    def run(self, options, data):
        "Correct for Nyquist ghosting due to field inhomogeneity."

        # Read the inhomogeneity field-map data from disk (Calculated using compute_fmap).
       
        # Loop over each phase-encode line of uncorrected data S'_mn.
           
            # Perform FFT of S'_mn with respect to n (frequency-encode direction) to
            # obtain S'_m(x).

            # Loop over each point x.

                # Calculate the perturbation kernel K_mm'(x) at x by: 
                # (1) Performing FFT with respect to y' of exp(i*2*pi*phi(x,y')*Delta t)
                #     and exp(i*2*pi*phi(x,y')*Delta t).
                # (2)  

                # Invert perturbation operator to obtain correction operator at x.

                # Apply correction operator at x to the distorted data S'_m(x) to
                # obtain the corrected data S_m(x).

            # Perform an inverse FFT in the read direction to obtain the corrected
            # k-space data S_mn.
