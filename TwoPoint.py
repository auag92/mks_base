import CorrMaster as CM

def TwoPoint(corrtype, cutoff, periodicity, *args):

    nargin = len(args)

    if periodicity == 'periodic':

        if corrtype = 'auto':
            if nargin == 1:
                # periodic 2pt Auto X
                H1 = args[0]
                GG = CM.corr_master('auto',cutoff, H1)
                GG = GG/H1.size

                return GG

            elif nargin == 2
                # periodic 2pt Auto Masked
                H1 = arg[0]
                M1 = arg[1]
                GG = CM.corr_master('auto',cutoff, H1*M1)
                BB = CM.corr_master('auto',cutoff, M1)
                GG = GG/BB

                return GG

        elif corrtype = 'cross':
            if nargin == 2:
                H1 = args[0]
                H2 = args[1]
                GG = CM.corr_master('cross',cutoff, H1, H2)
                GG = GG/H1.size

                return GG

            elif nargin == 3:
                % Periodic 2pt Cross Masked
                H1 = args[0]
                H2 = args[1]
                M1 = args[2]
                GG = CM.corr_master('cross',cutoff, H1*M1, H2*M1)
                BB = CM.corr_master('auto',cutoff, M1)
                GG = GG/BB

                return GG

            elif nargin == 4:
                % Periodic 2pt Cross Masked
                H1 = args[0]
                H2 = args[1]
                M1 = args[2]
                M2 = args[3]
                GG = CM.corr_master('cross',cutoff, H1*M1, H2*M2)
                BB = CM.corr_master('cross',cutoff, M1, M2)
                GG = GG/BB

                return GG
    elif periodicity == 'nonperiodic':
        if corrtype = 'auto':
            if nargin == 1:
                # periodic 2pt Auto X
                H1 = args[0]
                GG = CM.corr_master('auto',cutoff, H1)
                GG = GG/H1.size

                return GG

            elif nargin == 2
                # periodic 2pt Auto Masked
                H1 = arg[0]
                M1 = arg[1]
                GG = CM.corr_master('auto',cutoff, H1*M1)
                BB = CM.corr_master('auto',cutoff, M1)
                GG = GG/BB

                return GG

        elif corrtype = 'cross':
            if nargin == 2:
                H1 = args[0]
                H2 = args[1]
                GG = CM.corr_master('cross',cutoff, H1, H2)
                GG = GG/H1.size

                return GG

            elif nargin == 3:
                % Periodic 2pt Cross Masked
                H1 = args[0]
                H2 = args[1]
                M1 = args[2]
                GG = CM.corr_master('cross',cutoff, H1*M1, H2*M1)
                BB = CM.corr_master('auto',cutoff, M1)
                GG = GG/BB

                return GG

            elif nargin == 4:
                % Periodic 2pt Cross Masked
                H1 = args[0]
                H2 = args[1]
                M1 = args[2]
                M2 = args[3]
                GG = CM.corr_master('cross',cutoff, H1*M1, H2*M2)
                BB = CM.corr_master('cross',cutoff, M1, M2)
                GG = GG/BB

                return GG
