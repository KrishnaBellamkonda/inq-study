# pseudopod

Pseudopod is a header-only C++ library to provide pseudopotentials for density functional theory calculations to electronic structure codes.

It has the following functionality:

* Provide an interface for pseudopotential sets, so that users can directly request a set and a certain element without dealing with the file management associated.
* It bundles the pseudopotential sets that are available, so that they don't have to be replicated in the system and can be used by different codes and users.
* The library can provide the code with default values of energy cutoff and required simulation box size for converged calculations, so users can avoid the work of determining these values.
* The current sets included in the library are: pseudo-dojo, SG15 and HSCV. Other sets will be included in the future.
* It can parse pseudopotential files in several formats, providing the calling code with a uniform abstraction for a pseudopotential. This avoids the time consuming and error prone task of writing pseudopotential parsers.
* It support major pseudopotential formats like UPF (version 1 and 2), the QSO XML format from Qball, PSPML, and psp8. Other formats like Troullier-Martins and HGH will be supported in the future.
* The aim is to support all types of pseudopotentials. Norm-conserving and optimized-Vanderbilt norm-conserving are currently supported. Ultrasoft and PAWs will be supported in the future.
* It also provides another functionality required for pseudopotential application, like spherical harmonics, pseudopotential separation, filtering and interpolation.
