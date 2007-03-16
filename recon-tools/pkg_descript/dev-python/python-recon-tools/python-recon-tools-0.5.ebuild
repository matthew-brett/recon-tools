# Copyright 1999-2005 Gentoo Foundation
# Distributed under the terms of the GNU General Public License v2
# $Header: $

inherit distutils eutils

#MY_P=imaging-tools-${PV}
#S=${WORKDIR}/${MY_P}

DESCRIPTION="MRI reconstruction tools"
HOMEPAGE="https://cirl.berkeley.edu/view/BIC/ReconTools"
SRC_URI="https://cirl.berkeley.edu/twiki/pub/BIC/ReconTools/${P}.tar.gz"

LICENSE="as-is"
SLOT="0"
KEYWORDS="x86 ppc amd64 ia64 ppc64"
IUSE=""

DEPEND=">=dev-lang/python-2.4
        >=dev-python/numpy-1.0.1
	>=dev-python/matplotlib-0.87.7
	>=dev-python/pygtk-2.10"

src_install() {
	      distutils_src_install
	      distutils_python_version
}
