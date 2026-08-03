% EPOKA University Thesis LaTeX Class (CEN BA and MSc)
% Prepared by: Dr. Florenc Skuka
% Department of Computer Engineering, EPOKA University
% Based on the official EPOKA University Thesis Writing Guidelines

% This class provides a structured LaTeX template for preparing
% Bachelor (BA) and Master of Science (MSc) theses in the field of
% Computer Engineering Department.

% ------------------------------------------------------------------
% Version History
% ------------------------------------------------------------------
% v1.0 (2025/12)
% - Initial draft of the thesis template
% - Basic structure implemented (title page, chapters, TOC)
% - Preliminary formatting (spacing, margins, headings)
% - Known issues: inconsistencies in TOC formatting, appendix handling,
%   and heading styles

% v1.1 (2026/03)
% - Revised to align with official thesis writing guidelines
% - Corrected formatting inconsistencies (spacing, headings, numbering)
% - Improved structure of front matter (abstract, acknowledgments, lists)
% - Added support for Bachelor (BA – CEN) thesis format
% - General cleanup and stabilization of the template

% v1.2 (2026/04)
% - Fixed Table of Contents (TOC) formatting:
%     * Removed bold styling from TOC entries
%     * Ensured consistency with thesis formatting guidelines
% - Corrected Appendix representation:
%     * Replaced incorrect "Chapter A" with "Appendix A"
%     * Introduced "APPENDICES" section in TOC
% - Ensured References section is properly included in TOC
% - Minor refinements for consistency across sectionsarch 2026
% Updating ToC

% v1.3 (2026/07)
% - Aligned the generated PDF with the official Word template (compare/):
%     * Cover page rebuilt with Word's fixed, top-anchored line layout
%       (was a stretched full-page layout)
%     * Line spacing set to match Word's 1.5 lines (~0.72 cm) via \setstretch
%     * Approval sheet spacing matched to Word; declaration statement single-spaced
%     * Footer/page-number position lowered to ~1.8 cm from the bottom edge
%     * TOC/LOF/LOT titles lowered to sit just below the top margin; denser dots
%     * Acknowledgements heading spelled to match Word ("ACKNOWLEDGEMENTS")
%     * Albanian abstract now uses the Albanian title, degree and department
% - Approval date is no longer hardcoded: provide it via \approvaldate in metadata.tex


\NeedsTeXFormat{LaTeX2e}[1996/12/01]
\ProvidesClass{epoka}[2026/07/01 v1.3 Epoka University Thesis Class]

% Base class
\LoadClass[a4paper,oneside,onecolumn,12pt]{report}[1996/10/31]

% Required packages
\RequirePackage{ifthen}
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{fancyhdr}
\RequirePackage[hang,labelsep=period]{caption}
\RequirePackage{calc}
\RequirePackage{setspace}
\RequirePackage{geometry}
\RequirePackage{array}
\RequirePackage{longtable}
\RequirePackage{tocloft}
\RequirePackage{titlesec}
\RequirePackage{indentfirst}
\RequirePackage[nottoc,notlof,notlot]{tocbibind}
\RequirePackage{mathptmx}  % Times New Roman font

% Caption setup for tables and figures
\captionsetup[table]{position=top, aboveskip=24pt, belowskip=6pt}
\captionsetup[figure]{position=bottom, aboveskip=6pt, belowskip=18pt}

% Equation spacing
\AtBeginDocument{%
  \setlength{\abovedisplayskip}{24pt}%
  \setlength{\belowdisplayskip}{24pt}%
  \setlength{\abovedisplayshortskip}{24pt}%
  \setlength{\belowdisplayshortskip}{24pt}%
}

% Page geometry - Epoka University requirements
% Margins: Left 4cm, Right 2.5cm, Top 2.5cm, Bottom 2.5cm
\geometry{
    a4paper,
    left=4cm,
    right=2.5cm,
    top=2.5cm,
    bottom=2.5cm,
    headheight=15pt,
    headsep=10pt,
    footskip=0.75cm % Places the centered page number ~1.8 cm from the bottom edge (matches Word)
}

% Header and footer setup - page numbers bottom centered
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\thepage}  % Page number at bottom center
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

% Line spacing
% Word's "1.5 lines" spacing renders looser than LaTeX \onehalfspacing.
% \setstretch{1.42} reproduces Word's ~0.72 cm baseline-to-baseline for 12pt Times.
\setstretch{1.42}

% Paragraph formatting
\setlength{\parindent}{12mm}
\setlength{\parskip}{6pt}     % Default to 6pt (Abstract, Refs, Body) per PDF instructions

% Make figure and table captions bold and italic
\captionsetup{labelfont={bf,it}}

% Hyphenation penalties
\hyphenpenalty=5000
\exhyphenpenalty=5000

% Boolean flags
\newif\if@itsaMSc
\@itsaMSctrue  % Default is MSc

\newif\if@itsaBSc
\@itsaBScfalse

\newif\if@TabloListesiVar
\@TabloListesiVartrue

\newif\if@SekilListesiVar
\@SekilListesiVartrue

\newif\if@AbbreviationVar
\@AbbreviationVarfalse

\newif\if@DedicationVar
\@DedicationVarfalse

% Commands for thesis type
\newcommand*{\MSc}{\@itsaMSctrue \@itsaBScfalse}
\newcommand*{\PhD}{\@itsaMScfalse \@itsaBScfalse}
\newcommand*{\BSc}{\@itsaMScfalse \@itsaBSctrue}

% Storage commands
\newcommand{\th@StoreIn}[2]{\gdef#1{#2}}

% Author information
\renewcommand*{\author}[2]{%
    \th@StoreIn{\th@firstname}{#1}%
    \th@StoreIn{\th@lastname}{#2}%
    \th@StoreIn{\th@fullname}{#1 #2}%
}

% Thesis information
\renewcommand{\title}[1]{\th@StoreIn{\th@title}{#1}}
\newcommand{\titleenglish}[1]{\th@StoreIn{\th@titleenglish}{#1}}
\newcommand{\titlealbanian}[1]{\th@StoreIn{\th@titlealbanian}{#1}}
\renewcommand{\date}[1]{\th@StoreIn{\th@date}{#1}}
\newcommand{\dateenglish}[1]{\th@StoreIn{\th@dateenglish}{#1}}
% Approval date (used on the approval sheet). Defaults to empty if not set.
\newcommand{\th@approvaldate}{}
\newcommand{\approvaldate}[1]{\th@StoreIn{\th@approvaldate}{#1}}

% Supervisor information
\newcommand{\supervisor}[1]{\th@StoreIn{\th@supervisor}{#1}}
\newcommand{\supervisorenglish}[1]{\th@StoreIn{\th@supervisorenglish}{#1}}

% Department information
\newcommand{\department}[1]{\th@StoreIn{\th@department}{#1}}
\newcommand{\departmentenglish}[1]{\th@StoreIn{\th@departmentenglish}{#1}}
% Albanian department name (for the Albanian abstract). Falls back to
% \department if it is left unset.
\newcommand{\th@departmentalbanian}{}
\newcommand{\departmentalbanian}[1]{\th@StoreIn{\th@departmentalbanian}{#1}}
\newcommand{\headofdepartment}[1]{\th@StoreIn{\th@headofdepartment}{#1}}

% Faculty information
\newcommand{\faculty}[1]{\th@StoreIn{\th@faculty}{#1}}
\newcommand{\facultyenglish}[1]{\th@StoreIn{\th@facultyenglish}{#1}}


% Committee members
\newcommand{\committee}[5]{%
    \th@StoreIn{\th@committeeone}{#1}%
    \th@StoreIn{\th@committeetwo}{#2}%
    \th@StoreIn{\th@committeethree}{#3}%
    \th@StoreIn{\th@committeefour}{#4}%
    \th@StoreIn{\th@committeefive}{#5}%
}

% Front matter content
\renewcommand{\abstract}[1]{\th@StoreIn{\th@abstractcontent}{#1}}
\newcommand{\abstrakt}[1]{\th@StoreIn{\th@abstraktcontent}{#1}}
\newcommand{\acknowledgments}[1]{\th@StoreIn{\th@acknowledgments}{#1}}
\newcommand{\dedication}[1]{\@DedicationVartrue\th@StoreIn{\th@dedication}{#1}}
\newcommand{\abbreviations}[1]{\@AbbreviationVartrue\th@StoreIn{\th@abbreviations}{#1}}

% Keywords
\newcommand{\keywords}[1]{\th@StoreIn{\th@keywords}{#1}}
\newcommand{\keywordsenglish}[1]{\th@StoreIn{\th@keywordsenglish}{#1}}

% List control
\newcommand*{\NoTableList}{\@TabloListesiVarfalse}
\newcommand*{\NoFigureList}{\@SekilListesiVarfalse}

% Chapter formatting - 16pt font size
\titleformat{\chapter}[display]
    {\normalfont\bfseries\centering\fontsize{16}{19.2}\selectfont}
    {\MakeUppercase{\chaptertitlename}\ \thechapter}
    {20pt}
    {\bfseries\fontsize{16}{19.2}\selectfont\MakeUppercase}

\titlespacing*{\chapter}{0pt}{-24pt}{42pt} % -24pt before to ensure it starts high on the page, 42pt after (including 1.5 empty line + 6+6pt)

% Section formatting - with indentation
\titleformat{\section}
    {\normalfont\fontsize{14}{16.8}\selectfont\bfseries}
    {\thesection}
    {1em}
    {}
[\vspace{-0.5em}]  % Reduce space after section title

\titlespacing*{\section}{0.7cm}{48pt}{12pt}  % Left indent 7mm, 48pt before (incl 2 empty lines), 12pt after

% Subsection formatting - with indentation
\titleformat{\subsection}
    {\normalfont\fontsize{13}{15.6}\selectfont\bfseries}
    {\thesubsection}
    {1em}
    {}

\titlespacing*{\subsection}{0.7cm}{30pt}{6pt}  % Left indent 7mm, 30pt before (incl 1 empty line), 6pt after

% Subsubsection formatting - with indentation
\titleformat{\subsubsection}
    {\normalfont\bfseries}
    {\thesubsubsection}
    {1em}
    {}

\titlespacing*{\subsubsection}{0.7cm}{30pt}{6pt}  % Left indent 7mm, 30pt before, 6pt after

% Table of Contents formatting - 16pt centered
\renewcommand{\contentsname}{TABLE OF CONTENTS}
\renewcommand{\listfigurename}{LIST OF FIGURES}
\renewcommand{\listtablename}{LIST OF TABLES}
\renewcommand{\bibname}{REFERENCES}

% TOC depth
\setcounter{tocdepth}{3}
\setcounter{secnumdepth}{3}

% Standard TOC Entry formatting for chapters
\renewcommand{\cftchapfont}{\normalfont}
\renewcommand{\cftchappagefont}{\normalfont}
\renewcommand{\cftdot}{.}
\renewcommand{\cftdotsep}{0.5} % Dense dot leaders (matching Word template)
\renewcommand{\cftchapleader}{\cftdotfill{\cftdotsep}}
\setlength{\cftchapnumwidth}{0pt}
\setlength{\cftbeforechapskip}{12pt} 

% Section and Subsection TOC formatting
\setlength{\cftsecindent}{12pt}
\setlength{\cftsecnumwidth}{32pt}
\setlength{\cftbeforesecskip}{8pt} % Spacing for 1.5 lines feel
\setlength{\cftsubsecindent}{12pt}
\setlength{\cftsubsecnumwidth}{33pt}
\setlength{\cftbeforesubsecskip}{8pt}
\setlength{\cftsubsubsecindent}{12pt}
\setlength{\cftsubsubsecnumwidth}{42pt}
\setlength{\cftbeforesubsubsecskip}{8pt}

% Centering preliminary titles
\renewcommand{\cfttoctitlefont}{\hfill\fontsize{16}{19.2}\selectfont\bfseries}
\renewcommand{\cftaftertoctitle}{\hfill}
\renewcommand{\cftlottitlefont}{\hfill\fontsize{16}{19.2}\selectfont\bfseries}
\renewcommand{\cftafterlottitle}{\hfill}
\renewcommand{\cftloftitlefont}{\hfill\fontsize{16}{19.2}\selectfont\bfseries}
\renewcommand{\cftafterloftitle}{\hfill}

% Position TOC, LOF, LOT titles just below the top margin (matches Word)
\setlength{\cftbeforetoctitleskip}{-12pt}
\setlength{\cftbeforelottitleskip}{-12pt}
\setlength{\cftbeforeloftitleskip}{-12pt}

% Redefine internal chapter command to insert two lines in TOC correctly
\newif\if@isappendix
\@isappendixfalse

\renewcommand\appendix{\par
  \clearpage
  \setcounter{chapter}{0}%
  \setcounter{section}{0}%
  \gdef\@chapapp{Appendix}%
  \gdef\thechapter{\@Alph\c@chapter}%
  \@isappendixtrue
  \providecommand\phantomsection{}
  \phantomsection
  \addcontentsline{toc}{chapter}{APPENDICES}%
}

\def\@chapter[#1]#2{%
    \ifnum \c@secnumdepth >\m@ne
        \refstepcounter{chapter}%
        \typeout{\@chapapp\space\thechapter.}%
        \if@isappendix
            \addcontentsline{toc}{chapter}{Appendix \thechapter\space #1}%
        \else
            % Insert "CHAPTER X" line
            \addcontentsline{toc}{chapter}{CHAPTER \thechapter}%
            % Insert title line
            \addcontentsline{toc}{chapter}{#1}%
        \fi
    \else
        \addcontentsline{toc}{chapter}{#1}%
    \fi
    \chaptermark{#1}%
    \addtocontents{lof}{\protect\addvspace{10\p@}}%
    \addtocontents{lot}{\protect\addvspace{10\p@}}%
    \if@twocolumn
        \@topnewpage[\@makechapterhead{#2}]%
    \else
        \@makechapterhead{#2}%
        \@afterheading
    \fi
}

% Figure and table numbering
\renewcommand{\thefigure}{\thechapter.\arabic{figure}}
\renewcommand{\thetable}{\thechapter.\arabic{table}}
\renewcommand{\theequation}{\thechapter.\arabic{equation}}

% Title Page - Following Epoka University Format
% The Word template anchors the title near the top margin. The natural
% inter-block skips below are derived from the reference PDF, but they can
% shrink when a long title wraps. The final skip stretches so the date remains
% at the bottom of the same cover page.
\newcommand{\makecoverpage}{%
    \begin{titlepage}
        \centering
        % Reset line stretch so the fixed 19.7pt leading is exact
        \setstretch{1}\fontsize{12}{19.7}\selectfont
        % Anchor the title near the top of the text block while keeping all
        % cover-page content in one unbreakable page-height box.
        \vbox to \textheight{%
            \hsize=\textwidth
            \centering
            \setlength{\parindent}{0pt}%
            \setlength{\parskip}{0pt}%
            \vskip -21pt
            \MakeUppercase{\th@title}\par
            \vskip 87.7pt plus 0pt minus 87.7pt
            A THESIS SUBMITTED TO\par
            THE \MakeUppercase{\th@faculty}\par
            OF\par
            EPOKA UNIVERSITY\par
            \vskip 80.4pt plus 0pt minus 80.4pt
            BY\par
            \vskip 34.2pt plus 0pt minus 34.2pt
            \MakeUppercase{\th@fullname}\par
            \vskip 113.8pt plus 0pt minus 113.8pt
            IN PARTIAL FULFILLMENT OF THE REQUIREMENTS\par
        \if@itsaBSc
            FOR BACHELOR DEGREE\par
        \else
            FOR\par
            \if@itsaMSc
                THE DEGREE OF MASTER OF SCIENCE\par
            \else
                THE DEGREE OF DOCTOR OF PHILOSOPHY\par
            \fi
        \fi
            IN\par
            \MakeUppercase{\th@department}\par
            \vskip 107.3pt plus 1fill minus 107.3pt
            \MakeUppercase{\th@date}\par
        }%
    \end{titlepage}
    \clearpage
}

% Approval Page (Page i)
\newcommand{\makeapprovalpage}{%
    \thispagestyle{plain}
    \setlength{\parskip}{0pt}%
    \vspace*{-0.44cm}
    {\centering\textbf{Approval sheet of the Thesis}\par}

    \vspace{1.44cm}

    \noindent
    This is to certify that we have read this thesis entitled ``\textbf{\MakeUppercase{\th@titleenglish}}'' and that in our opinion it is fully adequate, in scope and quality, as a thesis for the degree of \if@itsaBSc Bachelor of Science\else\if@itsaMSc Master of Science\else Doctor of Philosophy\fi\fi.

    \vspace{1.30cm}

    {\singlespacing
    \begin{flushright}
        \rule{6cm}{0.4pt}\\[0.28cm]
        \th@headofdepartment\\
        Head of Department\\
        Date: \th@approvaldate
    \end{flushright}\par}

    \vspace{2.38cm}

    \noindent
    Examining Committee Members:\par

    \vspace{1.23cm}

    {\singlespacing
    \noindent
    \begin{tabular}{@{}p{0.45\textwidth}p{0.3\textwidth}l@{}}
        \th@committeeone & (\th@departmentenglish) & \rule{2.5cm}{0.4pt} \\[0.75cm]
        \th@committeetwo & (\th@departmentenglish) & \rule{2.5cm}{0.4pt} \\[0.75cm]
        \th@committeethree & (\th@departmentenglish) & \rule{2.5cm}{0.4pt} \\
    \end{tabular}\par}

    \clearpage
}

% Declaration Page (Page ii)
\newcommand{\makedeclarationpage}{%
    \thispagestyle{plain}

    {\singlespacing
    \noindent
    \textbf{I hereby declare that all information in this document has been obtained and presented in accordance with academic rules and ethical conduct. I also declare that, as required by these rules and conduct, I have fully cited and referenced all material and results that are not original to this work.}\par}

    \vspace{2.4cm}

    {\singlespacing
    \begin{flushright}
        \begin{tabular}{@{}rl@{}}
            Name Surname: & \th@fullname \\[1.75cm]
            Signature: & \rule{4cm}{0.4pt}
        \end{tabular}
    \end{flushright}\par}

    \clearpage
}

% Abstract page
\newcommand{\makeabstractpage}{%
    \chapter*{ABSTRACT}
    \addcontentsline{toc}{chapter}{ABSTRACT}
    
    \begin{center}
        \vspace*{-12pt} % Adjusting for chapter header space
        \vspace{\baselineskip}
        {\bfseries\fontsize{14}{16.8}\selectfont \MakeUppercase{\th@titleenglish}}\\[\baselineskip]
        \vspace{\baselineskip}
        \th@lastname, \th@firstname\\
        \if@itsaBSc B.Sc.,\else\if@itsaMSc M.Sc.,\else Ph.D.,\fi\fi\ Department of \th@departmentenglish\\
        Supervisor: \th@supervisorenglish\\[\baselineskip]
    \end{center}
    
    \th@abstractcontent
    
    \vspace{\baselineskip}
    \vspace{\baselineskip}
    \vspace{\baselineskip}
    
    \noindent
    \textbf{Keywords:} \textit{\th@keywordsenglish}
    
    \clearpage
}

% Albanian Abstract page
\newcommand{\makeabstraktpage}{%
    \chapter*{ABSTRAKT}
    \addcontentsline{toc}{chapter}{ABSTRAKT}
    
    \begin{center}
        \vspace*{-12pt}
        \vspace{\baselineskip}
        {\bfseries\fontsize{14}{16.8}\selectfont \MakeUppercase{\th@titlealbanian}}\\[\baselineskip]
        \vspace{\baselineskip}
        \th@lastname, \th@firstname\\
        \if@itsaBSc Bachelor,\else\if@itsaMSc Master Shkencor,\else Doktor i Shkencave,\fi\fi\ Departamenti i \ifx\th@departmentalbanian\@empty\th@department\else\th@departmentalbanian\fi\\
        Udhëheqësi: \th@supervisor\\[10pt]
    \end{center}
    
    \th@abstraktcontent
    
    \vspace{\baselineskip}
    \noindent
    \textbf{Fjalë kyçe:} \textit{\th@keywords}
    
    \clearpage
}

% Acknowledgments page (Epoka/Word template spells it "ACKNOWLEDGEMENTS")
\newcommand{\makeacknowledgments}{%
    \chapter*{\fontsize{16}{19.2}\selectfont\bfseries ACKNOWLEDGEMENTS}
    \addcontentsline{toc}{chapter}{ACKNOWLEDGEMENTS}

    \indent \th@acknowledgments

    \clearpage
}

% Dedication page (optional)
\newcommand{\makededication}{%
    \if@DedicationVar
        \thispagestyle{plain}
        \vspace*{5cm}
        
        \begin{center}
            \textit{\th@dedication}
        \end{center}
        
        \clearpage
    \fi
}

% Abbreviations page
\newcommand{\makeabbreviations}{%
    \if@AbbreviationVar
        \chapter*{LIST OF ABBREVIATIONS}
        \addcontentsline{toc}{chapter}{LIST OF ABBREVIATIONS}
        
        \th@abbreviations
        
        \clearpage
    \fi
}

% Front matter
\newcommand{\makefrontmatter}{%
    \pagenumbering{roman}
    \setcounter{page}{1}
    
    \makecoverpage
    \if@itsaBSc\else
        \makeapprovalpage
    \fi
    \makedeclarationpage
    \makeabstractpage
    \if@itsaBSc\else
        \makeabstraktpage
    \fi
    \makededication
    \makeacknowledgments
    \makeabbreviations
    
    % Table of Contents
    \begin{singlespace}
        \tableofcontents
    \end{singlespace}
    \clearpage
    
    % List of Tables (12pt spacing per guidelines)
    \if@TabloListesiVar
        {%
         \setlength{\parskip}{12pt}%
         \listoftables
        }
        \addcontentsline{toc}{chapter}{LIST OF TABLES}
        \clearpage
    \fi
    
    % List of Figures (12pt spacing per guidelines)
    \if@SekilListesiVar
        {%
         \setlength{\parskip}{12pt}%
         \listoffigures
        }
        \addcontentsline{toc}{chapter}{LIST OF FIGURES}
        \clearpage
    \fi
    
    % Start main matter
    \pagenumbering{arabic}
    \setcounter{page}{1}
}

% Bibliography setup
\bibliographystyle{plain}

\endinput
